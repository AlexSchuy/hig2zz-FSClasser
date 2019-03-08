import os
import re
from configparser import ConfigParser
from glob import glob

from config import settings


SAMPLE_TABLE = settings.get_setting('Final State Selection', 'sample_table')
V1_B_SAMPLE_DIR = os.path.join(
    '/', 'cefs', 'data', 'RecData', 'CEPC250', 'CEPC_v1', '4fermions')
V1_S_SAMPLE_DIR = os.path.join(
    '/', 'cefs', 'dirac', 'user', 'b', 'byzhang', 'higgs', 'rec')
V4_DIR = os.path.join('/', 'cefs', 'data', 'DstData', 'CEPC240', 'CEPC_v4')
V4_4F_DIR = os.path.join(V4_DIR, '4fermions')
V4_2F_DIR = os.path.join(V4_DIR, '2fermions')
V4_H_DIR = os.path.join(V4_DIR, 'higgs')
config = ConfigParser()
config.add_section('v1')
config.add_section('v4')

# higgs & 2 fermion
for root_dir in [V4_H_DIR, V4_2F_DIR]:
    for p in glob(os.path.join(root_dir, '*')):
        d = os.path.split(p)[1]
        if d in ['smart_final_states', 'qqh_X', 'nnh_X', 'e1e1h_X', 'e2e2h_X', 'e3e3h_X']:
            continue
        process = re.search('.P(.*).e0.p0', d).group(1)
        config.set('v4', process, os.path.join(p, '*'))

# sw_l samples
for key in ('sw_l0mu', 'sw_l0tau'):
    config.set('v4', key, os.path.join(
        V4_4F_DIR, 'E240.Psw_l.e0.p0.whizard195', key + '*'))

# sw_sl samples
for key in ('sw_sl0qq',):
    config.set('v4', key, os.path.join(
        V4_4F_DIR, 'E240.Psw_sl.e0.p0.whizard195', key + '*'))

# sze_l samples
for key in ('sze_l0e', 'sze_l0mu', 'sze_l0nunu', 'sze_l0tau'):
    config.set('v4', key, os.path.join(
        V4_4F_DIR, 'E240.Psze_l.e0.p0.whizard195', key + '*'))

# sze_sl samples
for key in ('sze_sl0dd', 'sze_sl0uu'):
    config.set('v4', key, os.path.join(
        V4_4F_DIR, 'E240.Psze_sl.e0.p0.whizard195', key + '*'))

for key, d in zip(('sze_sl0dd', 'sze_sl0uu'), ('dd', 'uu')):
    config.set('v1', key, os.path.join(V1_B_SAMPLE_DIR,
                                       'E250.Psze_sl.e0.p0.whizard195', d, '*'))

# szeorsw_l samples
config.set('v4', 'szeorsw_l0l', os.path.join(
    V4_4F_DIR, 'E240.Pszeorsw_l.e0.p0.whizard195', '*'))

# sznu_l samples
for key in ('sznu_l0mumu', 'sznu_l0tautau'):
    config.set('v4', key, os.path.join(
        V4_4F_DIR, 'E240.Psznu_l.e0.p0.whizard195', key + '*'))

# sznu_sl samples
for key in ('sznu_sl0nu_down', 'sznu_sl0nu_up'):
    config.set('v4', key, os.path.join(
        V4_4F_DIR, 'E240.Psznu_sl.e0.p0.whizard195', key + '*'))

# ww_h samples
for key in ('ww_h0ccbs', 'ww_h0ccds', 'ww_h0cuxx', 'ww_h0uubd', 'ww_h0uusd'):
    config.set('v4', key, os.path.join(
        V4_4F_DIR, 'E240.Pww_h.e0.p0.whizard195', key + '*'))

# ww_l samples
for key in ('ww_l0ll',):
    config.set('v4', key, os.path.join(
        V4_4F_DIR, 'E240.Pww_l.e0.p0.whizard195', key + '*'))

# ww_sl samples
for key in ('ww_sl0muq', 'ww_sl0tauq'):
    config.set('v4', key, os.path.join(
        V4_4F_DIR, 'E240.Pww_sl.e0.p0.whizard195', key + '*'))
    config.set('v1', key, os.path.join(V1_B_SAMPLE_DIR,
                                       'E250.Pww_sl.e0.p0.whizard195', key, '*'))

# zz_h samples
for key in ('zz_h0cc_nots', 'zz_h0dtdt', 'zz_h0utut', 'zz_h0uu_notd'):
    config.set('v4', key, os.path.join(
        V4_4F_DIR, 'E240.Pzz_h.e0.p0.whizard195', key + '*'))

# zz_l samples
for key in ('zz_l04mu', 'zz_l04tau', 'zz_l0mumu', 'zz_l0taumu', 'zz_l0tautau'):
    config.set('v4', key, os.path.join(
        V4_4F_DIR, 'E240.Pzz_l.e0.p0.whizard195', key + '*'))

for key, d in zip(('zz_l04mu', 'zz_l04tau', 'zz_l0mumu', 'zz_l0taumu', 'zz_l0tautau'), ('04mu', '04tau', 'mumu', 'taumu', 'tautau')):
    config.set('v1', key, os.path.join(V1_B_SAMPLE_DIR,
                                       'E250.Pzz_l.e0.p0.whizard195', d, '*.slcio'))

# zz_sl samples
for key in ('zz_sl0mu_down', 'zz_sl0mu_up', 'zz_sl0nu_down', 'zz_sl0nu_up', 'zz_sl0tau_down', 'zz_sl0tau_up'):
    config.set('v4', key, os.path.join(
        V4_4F_DIR, 'E240.Pzz_sl.e0.p0.whizard195', key + '*'))

# zzorww_h samples
for key in ('zzorww_h0cscs', 'zzorww_h0udud'):
    config.set('v4', key, os.path.join(
        V4_4F_DIR, 'E240.Pzzorww_h.e0.p0.whizard195', key + '*'))

# zzorww_l samples
for key in ('zzorww_l0mumu', 'zzorww_l0tautau'):
    config.set('v4', key, os.path.join(
        V4_4F_DIR, 'E240.Pzzorww_l.e0.p0.whizard195', key + '*'))

# v1 higgs
for key, filename in zip(('e1e1h', 'e2e2h', 'e3e3h', 'qqh', 'nnh', 'n2n2h_zz'), ('*e1e1h*', '*e2e2h*', '*e3e3h*', '*Pqqh*', '*nnh*', 'n2n2zz*')):
    config.set('v1', key, os.path.join(V1_S_SAMPLE_DIR, filename))

with open(SAMPLE_TABLE, 'wb+') as configfile:
    config.write(configfile)
