"""Client Portal: customers create VMs by cloning a template the hoster allows (twin of _create_ct).

The hoster's vm_create config fixes the cluster, node and storage. The customer only picks a
whitelisted template, cores, memory, disk and the Cloud-Init login. These tests check that:
  - only a whitelisted ID that is a real template on that node gets cloned,
  - the VMID comes from the tenant's range, and the quota check and the clone run inside the
    per-tenant lock,
  - bad input is refused as in _create_ct (a body that is not a JSON object, a password that is
    not a string or contains a NUL),
  - a VM that fails after the clone is deleted and gets no ACL.

test_pve_gets_the_login_and_the_log_does_not runs the real update_vm_config and _api_put and
fakes only the HTTP transport: PVE must get the password as typed and the SSH key URL-encoded,
and the manager's log line must contain neither.
"""
import contextlib
import inspect
import json
import logging
import types
from unittest.mock import MagicMock
from urllib.parse import parse_qs, quote

import pytest
import requests
from requests.adapters import BaseAdapter
from requests.models import Response

import plugins.client_portal as portal
from pegaprox.core.manager import PegaProxManager


KEY = 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIG2ZJIBiWHI6Ni8irVLL5jlM2abNk1r1dj9Bz+iWkA7D lab'
VM_FORM = {'template': 9000, 'hostname': 'web01', 'cores': 1, 'memory': 1024, 'disk_gb': 8,
           'username': 'cust', 'password': 'hunter2hunter2', 'ssh_key': KEY}
TEMPLATE_ROW = {'vmid': 9000, 'type': 'qemu', 'template': 1, 'node': 'pve1',
                'name': 'tpl-debian-13', 'maxdisk': 3 * 1024 ** 3}


def _cfg(target_cluster, templates=(9000,)):
    return {
        'allow_vm_create': True,
        'vm_create': {
            'cluster_id': target_cluster, 'node': 'pve1', 'storage': 'local-lvm',
            'templates': list(templates), 'full_clone': True,
            'max_cores': 2, 'max_memory_mb': 2048, 'max_disk_gb': 32,
        },
    }


@contextlib.contextmanager


def _as(api, username, body):
    from flask import request as _rq
    with api.app.test_request_context('/', base_url='http://localhost', json=body):
        _rq.session = {'user': username, 'role': 'user'}
        yield


def _ok(status=200, data=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = {'data': data}
    r.text = ''
    return r


class _Transport(BaseAdapter):
    """Fake HTTP transport: records every request and answers each one with a task ID."""

    def __init__(self):
        super().__init__()
        self.sent = []

    def send(self, request, **kw):
        self.sent.append(request)
        r = Response()
        r.status_code = 200
        r._content = json.dumps({'data': 'UPID:pve1:0000:test'}).encode()
        r.headers['Content-Type'] = 'application/json'
        r.request, r.url = request, request.url
        return r

    def close(self):
        pass


def _real_config_path(m):
    """Use the real update_vm_config and _api_put; only the HTTP transport is fake."""
    transport = _Transport()
    sess = requests.Session()
    sess.mount('https://', transport)
    m._create_session = lambda: sess
    m.api_timeout = 5
    m.logger = logging.getLogger('test.portal_vm_create.manager')
    m._api_put = types.MethodType(PegaProxManager._api_put, m)
    m.update_vm_config = types.MethodType(PegaProxManager.update_vm_config, m)
    return transport


@pytest.fixture
def estate(api, seed, monkeypatch):
    for cid in ('cluster_a', 'cluster_b'):
        seed.db.execute('''INSERT INTO clusters (id, name, host, user, pass_encrypted)
                           VALUES (?, ?, '10.0.0.1', 'root@pam', 'x')''', (cid, cid))
    seed.tenant('t_cust', ['cluster_a'])
    seed.user('cust', role='user', tenant_id='t_cust')

    made = {}
    for cid in ('cluster_a', 'cluster_b'):
        m = api.make_fake_manager(cid)
        m.is_connected = True
        m.host, m.api_port = '10.0.0.1', 8006
        m.get_vm_resources.return_value = [TEMPLATE_ROW]
        m.get_next_vmid.return_value = {'success': True, 'vmid': 105}
        m.clone_vm.return_value = {'success': True, 'data': 'UPID:pve1:clone'}
        m._wait_for_task.return_value = True
        m.update_vm_config.return_value = {'success': True}
        m._api_put.side_effect = lambda url, **kw: _ok(data='UPID:pve1:resize' if url.endswith('/resize') else None)
        api.set_manager(cid, m)
        made[cid] = m

    import pegaprox.utils.rbac as rbac
    monkeypatch.setattr(rbac, 'tenants_db', {}, raising=False)
    return made


# --- the target and the template --------------------------------------------------------------

def test_a_target_outside_the_tenant_is_refused(api, estate, monkeypatch):
    monkeypatch.setattr(portal, '_load_config', lambda: _cfg('cluster_b'))
    with _as(api, 'cust', VM_FORM):
        res = portal._create_vm()

    assert isinstance(res, tuple) and res[1] == 403, res
    estate['cluster_b'].clone_vm.assert_not_called()


def test_only_a_whitelisted_template_is_cloned(api, estate, monkeypatch):
    monkeypatch.setattr(portal, '_load_config', lambda: _cfg('cluster_a', templates=(9001,)))
    with _as(api, 'cust', VM_FORM):
        res = portal._create_vm()

    assert isinstance(res, tuple) and res[1] == 400, res
    estate['cluster_a'].clone_vm.assert_not_called()


def test_a_whitelisted_id_that_is_no_template_is_refused(api, estate, monkeypatch):
    """The whitelist holds IDs, so a normal VM that later reuses a whitelisted ID is refused."""
    estate['cluster_a'].get_vm_resources.return_value = [dict(TEMPLATE_ROW, template=0)]
    monkeypatch.setattr(portal, '_load_config', lambda: _cfg('cluster_a'))
    with _as(api, 'cust', VM_FORM):
        res = portal._create_vm()

    assert isinstance(res, tuple) and res[1] == 400, res
    estate['cluster_a'].clone_vm.assert_not_called()


def test_a_disk_smaller_than_the_template_is_refused(api, estate, monkeypatch):
    """PVE accepts a shrink and only fails it in the task, so it has to be refused up front."""
    monkeypatch.setattr(portal, '_load_config', lambda: _cfg('cluster_a'))
    with _as(api, 'cust', dict(VM_FORM, disk_gb=2)):
        res = portal._create_vm()

    assert isinstance(res, tuple) and res[1] == 400, res
    estate['cluster_a'].clone_vm.assert_not_called()


# --- what the customer sends ------------------------------------------------------------------

@pytest.mark.parametrize('body', [[VM_FORM], 'web01', 5], ids=['list', 'string', 'number'])
def test_a_body_that_is_no_object_is_refused(api, estate, monkeypatch, body):
    monkeypatch.setattr(portal, '_load_config', lambda: _cfg('cluster_a'))
    with _as(api, 'cust', body):
        res = portal._create_vm()

    assert isinstance(res, tuple) and res[1] == 400, res
    assert res[0]['error'] == 'Expected a JSON object', res
    estate['cluster_a'].clone_vm.assert_not_called()


@pytest.mark.parametrize('password', [12345678, 1234567.5, ['a'] * 8, {str(i): i for i in range(8)}, True],
                         ids=['int', 'float', 'list', 'dict', 'bool'])
def test_a_password_that_is_no_string_is_refused(api, estate, monkeypatch, password):
    monkeypatch.setattr(portal, '_load_config', lambda: _cfg('cluster_a'))
    with _as(api, 'cust', dict(VM_FORM, password=password)):
        res = portal._create_vm()

    assert isinstance(res, tuple) and res[1] == 400, res
    assert res[0]['error'] == 'Password must be a string', res
    estate['cluster_a'].clone_vm.assert_not_called()


@pytest.mark.parametrize('password', ['\x00' * 8, 'hunter2\x00hunter2', 'hunter2hunter2\x00'],
                         ids=['only_nul', 'nul_inside', 'nul_at_end'])
def test_a_password_with_a_nul_is_refused(api, estate, monkeypatch, password):
    """PVE's crypt() stops at a NUL, so the guest would end up with a shorter password."""
    monkeypatch.setattr(portal, '_load_config', lambda: _cfg('cluster_a'))
    with _as(api, 'cust', dict(VM_FORM, password=password)):
        res = portal._create_vm()

    assert isinstance(res, tuple) and res[1] == 400, res
    assert res[0]['error'] == 'Password must not contain a NUL character', res
    estate['cluster_a'].clone_vm.assert_not_called()


# --- the clone ---------------------------------------------------------------------------------

def test_the_vm_is_cloned_configured_and_granted(api, estate, monkeypatch):
    monkeypatch.setattr(portal, '_load_config', lambda: _cfg('cluster_a'))
    with _as(api, 'cust', VM_FORM):
        res = portal._create_vm()

    assert not isinstance(res, tuple), res
    assert res.get('success') is True and res['vmid'] == 105, res
    m = estate['cluster_a']
    m.clone_vm.assert_called_once()
    args, kw = m.clone_vm.call_args
    assert args[:4] == ('pve1', 9000, 'qemu', 105), args
    assert kw.get('full') is True and kw.get('target_storage') == 'local-lvm', kw

    m.update_vm_config.assert_called_once()
    args = m.update_vm_config.call_args.args
    assert args[:3] == ('pve1', 105, 'qemu'), args
    sent = args[3]
    assert sent['cores'] == 1 and sent['memory'] == 1024 and sent['ciuser'] == 'cust'
    assert ' ' not in sent['sshkeys'] and sent['sshkeys'].startswith('ssh-ed25519%20'), sent['sshkeys']

    resizes = [c for c in m._api_put.call_args_list if c.args[0].endswith('/qemu/105/resize')]
    assert resizes and resizes[0].kwargs['data'] == {'disk': 'scsi0', 'size': '8G'}

    from pegaprox.utils.rbac import load_vm_acls, invalidate_vm_acls_cache
    invalidate_vm_acls_cache()
    assert 'cust' in (load_vm_acls().get('cluster_a', {}).get('105') or {}).get('users', [])


def test_pve_gets_the_login_and_the_log_does_not(api, estate, monkeypatch, caplog):
    transport = _real_config_path(estate['cluster_a'])
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(portal, '_load_config', lambda: _cfg('cluster_a'))
    with _as(api, 'cust', VM_FORM):
        res = portal._create_vm()

    assert not isinstance(res, tuple) and res.get('success') is True, res
    [sent] = [parse_qs(r.body or '') for r in transport.sent
              if r.method == 'PUT' and r.url.endswith('/qemu/105/config')]
    assert sent['cipassword'] == [VM_FORM['password']]
    assert sent['sshkeys'] == [quote(KEY, safe='')]
    logged = '\n'.join(r.getMessage() for r in caplog.records)
    assert 'Updating qemu/105 config' in logged, logged
    assert VM_FORM['password'] not in logged, logged
    assert 'AAAAC3NzaC1lZDI1NTE5' not in logged, logged


def test_the_vmid_comes_from_the_tenant_range(api, estate, seed, monkeypatch):
    seed.db.save_tenant('t_cust', {'name': 't_cust', 'clusters': ['cluster_a'],
                                   'vmid_range_start': 1000, 'vmid_range_end': 1099})
    estate['cluster_a'].get_vm_resources.return_value = [TEMPLATE_ROW, {'vmid': 1000, 'type': 'qemu'}]
    monkeypatch.setattr(portal, '_load_config', lambda: _cfg('cluster_a'))
    with _as(api, 'cust', VM_FORM):
        res = portal._create_vm()

    assert not isinstance(res, tuple), res
    assert res['vmid'] == 1001, res
    assert estate['cluster_a'].clone_vm.call_args.args[3] == 1001


def test_a_failed_config_removes_the_clone_and_grants_nothing(api, estate, monkeypatch):
    m = estate['cluster_a']
    m.update_vm_config.return_value = {'success': False, 'error': 'refused'}
    monkeypatch.setattr(portal, '_load_config', lambda: _cfg('cluster_a'))
    with _as(api, 'cust', VM_FORM):
        res = portal._create_vm()

    assert isinstance(res, tuple) and res[1] == 500, res
    m.delete_vm.assert_called_once()
    assert m.delete_vm.call_args.args[:3] == ('pve1', 105, 'qemu')
    from pegaprox.utils.rbac import load_vm_acls, invalidate_vm_acls_cache
    invalidate_vm_acls_cache()
    assert '105' not in load_vm_acls().get('cluster_a', {})


def test_a_resize_that_times_out_still_removes_the_clone(api, estate, monkeypatch):
    """_api_put re-raises a timeout instead of answering; the cleanup must still remove the VM."""
    m = estate['cluster_a']
    m._api_put.side_effect = requests.exceptions.Timeout('PVE did not answer')
    monkeypatch.setattr(portal, '_load_config', lambda: _cfg('cluster_a'))
    with _as(api, 'cust', VM_FORM):
        res = portal._create_vm()

    assert isinstance(res, tuple) and res[1] == 500, res
    m.delete_vm.assert_called_once()
    from pegaprox.utils.rbac import load_vm_acls, invalidate_vm_acls_cache
    invalidate_vm_acls_cache()
    assert '105' not in load_vm_acls().get('cluster_a', {})


def test_the_check_and_the_clone_are_inside_the_tenant_lock():
    """Reads the source, like test_portal_quota_race.py does for containers: two requests at
    once must not both pass the quota check."""
    body = inspect.getsource(portal._create_vm)
    stripped = '\n'.join(l for l in body.split('\n') if not l.strip().startswith('#'))
    i_lock = stripped.index('_tenant_create_lock(')
    i_check = stripped.index('check_tenant_quota(')
    i_clone = stripped.index('clone_vm(')
    assert i_lock < i_check < i_clone
