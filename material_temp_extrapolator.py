"""
Abaqus Material Interpolation/Extrapolation Utility (FINAL)
-----------------------------------------------------------
Atualizações implementadas (para resolver o "flatline" / cauda constante):
- Adicionado controle explícito do domínio de eps_p quando curvas *PLASTIC têm comprimentos diferentes:
    EPS_DOMAIN_MODE:
      - "intersection": usa domínio comum (interseção) entre as temperaturas usadas no método -> recomendado
      - "reference": usa o eps da curva de referência (mais próxima de T_TARGET), truncado ao domínio comum

- Adicionado controle de extrapolação em eps_p ao interpolar sigma(eps):
    EPS_EXTRAP_MODE:
      - "nan": fora do range vira NaN (e depois é mascarado/truncado) -> recomendado junto com truncagem
      - "clamp": comportamento antigo (np.interp clamp) -> gera flatline
      - "linear": extrapola em eps usando a inclinação do primeiro/último trecho

- M1 (linear local em T) agora:
    * calcula apenas com os 2 pontos bracketing de T_TARGET
    * gera eps_target no domínio comum dessas 2 temperaturas
    * interpola sigma(eps) em cada temperatura sem clamp
    * interpola em T para obter sigma_target

- M2 (quadrático local em T) agora:
    * usa 3 temperaturas mais próximas de T_TARGET
    * gera eps_target no domínio comum dessas 3 temperaturas
    * interpola sigma(eps) em cada temperatura sem clamp
    * faz ajuste quadrático em T ponto-a-ponto para obter sigma_target

- M3 (scale by yield) permanece igual (não gera cauda porque usa eps da curva ref)

Observações:
- As curvas originais são plotadas como estão no input.
- As curvas geradas (M1/M2/M3) agora não "estouram" o eixo x nem criam platôs artificiais,
  a menos que você escolha EPS_EXTRAP_MODE="clamp" ou "linear" e/ou use um eps_grid maior.

Requisitos:
- Python 3.x
- numpy, matplotlib
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 0) CONFIGURAÇÕES DO USUÁRIO
# =============================================================================

# --- Cole aqui sua carta do Abaqus ---
MATERIAL_TEXT = r"""
*MATERIAL, NAME=TRC103P
*DENSITY
                 1.02E-9,
*EXPANSION, TYPE=ISO
                   8.E-5,
*ELASTIC, TYPE=ISOTROPIC
                   1417.,                     0.35,                      23.
                   1303.,                     0.35,                      40.
                   1133.,                     0.35,                      60.
                    921.,                     0.35,                      80.
                    638.,                     0.35,                     100.
*PLASTIC
                  17.204,                       0.,                      23.
                  18.025,              0.016835273,                      23.
                   19.08,              0.044800673,                      23.
                   20.35,              0.080945474,                      23.
                    23.4,               0.16580391,                      23.
                   30.75,              0.383759226,                      23.
                     43.,              0.662794239,                      23.
                15.82768,                       0.,                      40.
                  16.583,              0.016835273,                      40.
                 17.5536,              0.044800673,                      40.
                  18.722,              0.080945474,                      40.
                  21.528,               0.16580391,                      40.
                   28.29,              0.383759226,                      40.
                   39.56,              0.662794239,                      40.
                 13.7632,                       0.,                      60.
                   14.42,              0.016835273,                      60.
                  15.264,              0.044800673,                      60.
                   16.28,              0.080945474,                      60.
                   18.72,               0.16580391,                      60.
                    24.6,              0.383759226,                      60.
                    34.4,              0.662794239,                      60.
                 11.1826,                       0.,                      80.
                 11.7214,              0.016827376,                      80.
                  12.402,              0.044798235,                      80.
                  13.233,                0.0809369,                      80.
                  15.216,              0.165794403,                      80.
                  19.995,               0.38374715,                      80.
                   27.94,              0.662799606,                      80.
                  7.7418,                       0.,                     100.
                  8.1164,              0.016830522,                     100.
                   8.586,              0.044804193,                     100.
                   9.163,              0.080940603,                     100.
                  10.536,              0.165798818,                     100.
                   13.83,              0.383776661,                     100.
                   19.36,              0.662786491,                     100.
"""

# --- Temperatura alvo (pode ser qualquer valor) ---
T_TARGET = -30.0

# --- Exportação / Plot ---
EXPORT_FIGURES = False
SHOW_FIGURES = True
EXPORT_CARDS = True

# --- Pasta/Prefixos de saída ---
OUTPUT_PREFIX = "OUT_"

# --- Método M3 (shape) ---
SCALE_REF_T: str | float = "closest"  # ou 23.0, 40.0, etc.

# --- Tolerâncias ---
EPS_MATCH_TOL = 1e-9

# --- Controle do domínio de eps_p quando grids têm comprimentos diferentes ---
# "intersection" -> usar domínio comum das temperaturas usadas no método (mais defensável)
# "reference"    -> usar eps da curva de referência (closest a T_TARGET) truncado ao domínio comum
EPS_DOMAIN_MODE = "reference"

# --- Como tratar extrapolação em eps_p no mapeamento sigma(eps) ---
# "nan"   -> fora do range vira NaN (e é mascarado) [recomendado com truncagem]
# "clamp" -> platô (comportamento np.interp padrão) [gera flatline]
# "linear"-> extrapola em eps usando inclinação do primeiro/último trecho
EPS_EXTRAP_MODE = "nan"

# --- Quantidade de pontos do eps_grid quando usar "intersection" (grid artificial) ---
EPS_INTERSECTION_NPTS = 250


# =============================================================================
# 1) ESTRUTURAS / UTILITÁRIOS
# =============================================================================

@dataclass
class ElasticData:
    temps: np.ndarray
    E: np.ndarray
    nu: float  # assumido constante


@dataclass
class PlasticData:
    temps: np.ndarray
    eps_grid: np.ndarray          # grid comum de eps_p (apenas para utilidades gerais)
    sigma_mat: np.ndarray         # shape (nT, nEps) (apenas para utilidades gerais)
    original_curves: Dict[float, Tuple[np.ndarray, np.ndarray]]  # {T: (eps, sigma)}


def _clean_line(s: str) -> str:
    return s.strip().rstrip(",")


def _is_keyword(line: str) -> bool:
    return line.strip().startswith("*")


def _parse_floats_from_csv_line(line: str) -> List[float]:
    parts = [p.strip() for p in _clean_line(line).split(",") if p.strip()]
    return [float(p) for p in parts]


def _enforce_non_decreasing(y: np.ndarray) -> np.ndarray:
    """Enforce monotonic non-decreasing array."""
    out = y.copy()
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def _safe_positive(y: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    return np.maximum(y, floor)


def _find_bracketing_indices(x_sorted: np.ndarray, x0: float) -> Tuple[int, int, str]:
    """
    Returns (i_low, i_high, region) where region in {"below", "inside", "above"}.
    Assumes x_sorted strictly increasing.
    """
    if x0 <= x_sorted[0]:
        return 0, 1 if len(x_sorted) > 1 else 0, "below"
    if x0 >= x_sorted[-1]:
        return (len(x_sorted) - 2 if len(x_sorted) > 1 else 0), len(x_sorted) - 1, "above"

    i_high = int(np.searchsorted(x_sorted, x0, side="right"))
    i_low = i_high - 1
    return i_low, i_high, "inside"


def piecewise_linear_in_T(T: np.ndarray, y: np.ndarray, T_target: float) -> float:
    """
    Local piecewise linear in temperature:
    - inside range: interpolation between bracketing points
    - outside range: linear extrap using first two / last two points
    """
    T = np.asarray(T, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(T) != len(y):
        raise ValueError("T and y must have same length.")
    if len(T) < 2:
        return float(y[0])

    i0, i1, _ = _find_bracketing_indices(T, T_target)
    T0, T1 = T[i0], T[i1]
    y0, y1 = y[i0], y[i1]
    if np.isclose(T1, T0):
        return float(y0)
    w = (T_target - T0) / (T1 - T0)
    return float(y0 + w * (y1 - y0))


def local_quadratic_in_T(T: np.ndarray, y: np.ndarray, T_target: float) -> float:
    """
    Quadrático LOCAL (3 pontos mais próximos de T_target).
    Se houver <3 pontos: cai para linear local.
    """
    T = np.asarray(T, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(T)
    if n < 3:
        return piecewise_linear_in_T(T, y, T_target)

    idx = np.argsort(np.abs(T - T_target))[:3]
    T3 = T[idx]
    y3 = y[idx]
    p = np.polyfit(T3, y3, deg=2)
    return float(np.polyval(p, T_target))


def _choose_ref_temperature(temps: np.ndarray, rule: str | float, T_target: float) -> float:
    if isinstance(rule, (int, float)):
        return float(rule)
    rule = str(rule).strip().lower()
    if rule == "closest":
        return float(temps[int(np.argmin(np.abs(temps - T_target)))])
    raise ValueError("SCALE_REF_T inválido. Use 'closest' ou um número (ex.: 23.0).")


def interp_sigma_vs_eps(eps_new: np.ndarray, eps: np.ndarray, sig: np.ndarray, mode: str) -> np.ndarray:
    """
    Interpola sigma(eps) e controla o comportamento fora do domínio.
    mode:
      - "nan"   -> NaN fora do range
      - "clamp" -> platô (extremos)
      - "linear"-> extrapolação linear usando inclinação do 1º e do último trecho
    """
    eps = np.asarray(eps, float)
    sig = np.asarray(sig, float)
    eps_new = np.asarray(eps_new, float)

    # NaN fora do domínio
    out = np.interp(eps_new, eps, sig, left=np.nan, right=np.nan)

    mode = str(mode).strip().lower()
    if mode == "nan":
        return out

    if mode == "clamp":
        out2 = out.copy()
        out2 = np.where(np.isnan(out2) & (eps_new < eps[0]), sig[0], out2)
        out2 = np.where(np.isnan(out2) & (eps_new > eps[-1]), sig[-1], out2)
        return out2

    if mode == "linear":
        if len(eps) < 2:
            # sem informação de inclinação -> clamp
            return np.where(np.isnan(out), sig[0], out)

        out2 = out.copy()

        # abaixo
        denom0 = (eps[1] - eps[0])
        m0 = (sig[1] - sig[0]) / denom0 if denom0 != 0 else 0.0
        mask_lo = np.isnan(out2) & (eps_new < eps[0])
        out2[mask_lo] = sig[0] + m0 * (eps_new[mask_lo] - eps[0])

        # acima
        denom1 = (eps[-1] - eps[-2])
        m1 = (sig[-1] - sig[-2]) / denom1 if denom1 != 0 else 0.0
        mask_hi = np.isnan(out2) & (eps_new > eps[-1])
        out2[mask_hi] = sig[-1] + m1 * (eps_new[mask_hi] - eps[-1])

        return out2

    raise ValueError("EPS_EXTRAP_MODE inválido. Use: 'nan', 'clamp' ou 'linear'.")


def _domain_common_eps(temps_used: List[float], pl: PlasticData) -> Tuple[float, float]:
    """Retorna (eps_min, eps_max) do domínio comum entre as temperaturas informadas."""
    eps_mins = []
    eps_maxs = []
    for T in temps_used:
        eps_i, _ = pl.original_curves[float(T)]
        eps_mins.append(float(np.min(eps_i)))
        eps_maxs.append(float(np.max(eps_i)))
    return max(eps_mins), min(eps_maxs)


def _build_eps_target(pl: PlasticData, temps_used: List[float], T_target: float) -> np.ndarray:
    """
    Constrói o eps_target:
      - truncado ao domínio comum das temps_used
      - de acordo com EPS_DOMAIN_MODE
    """
    eps_min, eps_max = _domain_common_eps(temps_used, pl)
    if eps_max <= eps_min:
        raise ValueError(
            f"Domínio comum de eps_p inválido (eps_max <= eps_min) para temps_used={temps_used}. "
            "As curvas podem não se sobrepor em eps_p."
        )

    mode = str(EPS_DOMAIN_MODE).strip().lower()
    if mode == "intersection":
        return np.linspace(eps_min, eps_max, int(EPS_INTERSECTION_NPTS), dtype=float)

    if mode == "reference":
        # referência: curva mais próxima de T_target (dentre as disponíveis)
        Tref = float(pl.temps[int(np.argmin(np.abs(pl.temps - T_target)))])
        eps_ref, _ = pl.original_curves[Tref]
        eps_ref = np.asarray(eps_ref, dtype=float)
        eps_ref = eps_ref[(eps_ref >= eps_min) & (eps_ref <= eps_max)]
        if len(eps_ref) < 2:
            # fallback para intersection
            return np.linspace(eps_min, eps_max, int(EPS_INTERSECTION_NPTS), dtype=float)
        return eps_ref.copy()

    raise ValueError("EPS_DOMAIN_MODE inválido. Use: 'intersection' ou 'reference'.")


# =============================================================================
# 2) PARSER DA CARTA ABAQUS (preservando textos antes/depois dos blocos)
# =============================================================================

@dataclass
class MaterialSections:
    header_lines: List[str]        # tudo antes de *ELASTIC
    elastic_kw_line: str           # linha *ELASTIC...
    elastic_data_lines: List[str]  # linhas numéricas do elastic
    middle_lines: List[str]        # tudo entre fim do elastic e começo do plastic
    plastic_kw_line: str           # linha *PLASTIC...
    plastic_data_lines: List[str]  # linhas numéricas do plastic
    footer_lines: List[str]        # tudo depois do plastic


def split_material_sections(text: str) -> MaterialSections:
    lines = [l.rstrip("\n") for l in text.splitlines() if l.strip()]

    elastic_idx = None
    plastic_idx = None
    for i, l in enumerate(lines):
        u = l.strip().upper()
        if u.startswith("*ELASTIC"):
            elastic_idx = i
        if u.startswith("*PLASTIC"):
            plastic_idx = i

    if elastic_idx is None or plastic_idx is None:
        raise ValueError("Não encontrei *ELASTIC e/ou *PLASTIC no MATERIAL_TEXT.")
    if plastic_idx < elastic_idx:
        raise ValueError("*PLASTIC apareceu antes de *ELASTIC (inesperado).")

    header = lines[:elastic_idx]
    elastic_kw = lines[elastic_idx]
    elastic_data = []
    i = elastic_idx + 1
    while i < len(lines) and not _is_keyword(lines[i]):
        elastic_data.append(lines[i])
        i += 1

    middle = lines[i:plastic_idx]
    plastic_kw = lines[plastic_idx]
    plastic_data = []
    j = plastic_idx + 1
    while j < len(lines) and not _is_keyword(lines[j]):
        plastic_data.append(lines[j])
        j += 1

    footer = lines[j:]

    return MaterialSections(
        header_lines=header,
        elastic_kw_line=elastic_kw,
        elastic_data_lines=elastic_data,
        middle_lines=middle,
        plastic_kw_line=plastic_kw,
        plastic_data_lines=plastic_data,
        footer_lines=footer,
    )


def parse_elastic(elastic_data_lines: List[str]) -> ElasticData:
    T_list = []
    E_list = []
    nu_list = []
    for l in elastic_data_lines:
        vals = _parse_floats_from_csv_line(l)
        if len(vals) < 3:
            continue
        E_list.append(vals[0])
        nu_list.append(vals[1])
        T_list.append(vals[2])

    if len(T_list) == 0:
        raise ValueError("Bloco *ELASTIC vazio ou inválido.")

    temps = np.array(T_list, dtype=float)
    E = np.array(E_list, dtype=float)
    nu_arr = np.array(nu_list, dtype=float)

    idx = np.argsort(temps)
    temps = temps[idx]
    E = E[idx]
    nu_arr = nu_arr[idx]

    nu = float(nu_arr[0])
    if np.max(np.abs(nu_arr - nu)) > 1e-6:
        print("[WARN] ν varia com T no input; o script vai assumir ν = primeiro valor.")

    return ElasticData(temps=temps, E=E, nu=nu)


def parse_plastic(plastic_data_lines: List[str]) -> PlasticData:
    curves: Dict[float, List[Tuple[float, float]]] = {}
    for l in plastic_data_lines:
        vals = _parse_floats_from_csv_line(l)
        if len(vals) < 3:
            continue
        sigma, eps_p, T = float(vals[0]), float(vals[1]), float(vals[2])
        curves.setdefault(T, []).append((eps_p, sigma))

    if len(curves) == 0:
        raise ValueError("Bloco *PLASTIC vazio ou inválido.")

    original: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    for T, pts in curves.items():
        pts = sorted(pts, key=lambda x: x[0])
        eps = np.array([p[0] for p in pts], dtype=float)
        sig = np.array([p[1] for p in pts], dtype=float)
        original[float(T)] = (eps, sig)

    temps = np.array(sorted(original.keys()), dtype=float)

    # construir eps_grid "global" apenas para utilidades (M3 usa para σy)
    eps_base = original[float(temps[0])][0]
    same_grid = True
    for T in temps[1:]:
        eps_i = original[float(T)][0]
        if len(eps_i) != len(eps_base) or np.max(np.abs(eps_i - eps_base)) > EPS_MATCH_TOL:
            same_grid = False
            break

    if same_grid:
        eps_grid = eps_base.copy()
    else:
        # grid global = união de todos os eps, para permitir interpolação estável e σy(T)
        eps_all = np.concatenate([original[float(T)][0] for T in temps])
        eps_grid = np.unique(np.sort(eps_all))

    sigma_mat = np.zeros((len(temps), len(eps_grid)), dtype=float)
    for i, T in enumerate(temps):
        eps_i, sig_i = original[float(T)]
        # aqui, por ser "global", vamos usar clamp (np.interp) apenas para preencher sigma_mat.
        # Esse sigma_mat NÃO será usado para gerar M1/M2 (para evitar flatline).
        sigma_mat[i, :] = np.interp(eps_grid, eps_i, sig_i)

    for i in range(len(temps)):
        sigma_mat[i, :] = _enforce_non_decreasing(_safe_positive(sigma_mat[i, :]))

    return PlasticData(temps=temps, eps_grid=eps_grid, sigma_mat=sigma_mat, original_curves=original)


# =============================================================================
# 3) GERAR PROPRIEDADES EM T_TARGET (3 MÉTODOS)
# =============================================================================

@dataclass
class MethodResult:
    tag: str
    E_target: float
    nu: float
    eps_target: np.ndarray
    sigma_target: np.ndarray


def compute_target_linear(el: ElasticData, pl: PlasticData, T_target: float) -> MethodResult:
    """
    M1: Linear local em T usando APENAS as 2 temperaturas bracketing de T_target.
    eps_target é truncado ao domínio comum dessas 2 curvas -> evita "flatline".
    """
    E_t = piecewise_linear_in_T(el.temps, el.E, T_target)

    i0, i1, _ = _find_bracketing_indices(pl.temps, T_target)
    T0 = float(pl.temps[i0])
    T1 = float(pl.temps[i1])

    temps_used = [T0, T1]
    eps_target = _build_eps_target(pl, temps_used, T_target)

    eps0, sig0 = pl.original_curves[T0]
    eps1, sig1 = pl.original_curves[T1]

    s0 = interp_sigma_vs_eps(eps_target, eps0, sig0, mode=EPS_EXTRAP_MODE)
    s1 = interp_sigma_vs_eps(eps_target, eps1, sig1, mode=EPS_EXTRAP_MODE)

    mask = ~np.isnan(s0) & ~np.isnan(s1)
    eps_target = eps_target[mask]
    s0 = s0[mask]
    s1 = s1[mask]

    if np.isclose(T1, T0):
        sigma_t = s0
    else:
        w = (T_target - T0) / (T1 - T0)
        sigma_t = s0 + w * (s1 - s0)

    sigma_t = _enforce_non_decreasing(_safe_positive(sigma_t))

    return MethodResult(
        tag="M1_LINEAR_LOCAL",
        E_target=float(max(E_t, 1e-12)),
        nu=el.nu,
        eps_target=eps_target,
        sigma_target=sigma_t,
    )


def compute_target_local_quadratic(el: ElasticData, pl: PlasticData, T_target: float) -> MethodResult:
    """
    M2: Quadrático local em T usando as 3 temperaturas mais próximas de T_target.
    eps_target é truncado ao domínio comum dessas 3 curvas -> evita "flatline".
    """
    E_t = local_quadratic_in_T(el.temps, el.E, T_target)

    if len(pl.temps) < 3:
        # fallback direto para linear
        return compute_target_linear(el, pl, T_target)._replace(tag="M2_QUADRATIC_LOCAL_FALLBACK")

    idx3 = np.argsort(np.abs(pl.temps - T_target))[:3]
    T_used = [float(pl.temps[i]) for i in idx3]

    eps_target = _build_eps_target(pl, T_used, T_target)

    # calcular sigma(eps) nas 3 temperaturas e ajustar quadrático em T ponto-a-ponto
    sig_stack = []
    for T in T_used:
        eps_i, sig_i = pl.original_curves[float(T)]
        s_i = interp_sigma_vs_eps(eps_target, eps_i, sig_i, mode=EPS_EXTRAP_MODE)
        sig_stack.append(s_i)

    sig_stack = np.vstack(sig_stack)  # (3, nEps)
    # mascarar pontos onde alguma temperatura deu NaN (não deveria se truncou bem, mas por segurança)
    mask = np.all(~np.isnan(sig_stack), axis=0)
    eps_target = eps_target[mask]
    sig_stack = sig_stack[:, mask]

    # ajuste quadrático em T para cada coluna
    T3 = np.array(T_used, dtype=float)
    sigma_t = np.zeros(sig_stack.shape[1], dtype=float)
    for j in range(sig_stack.shape[1]):
        y3 = sig_stack[:, j]
        p = np.polyfit(T3, y3, deg=2)
        sigma_t[j] = float(np.polyval(p, T_target))

    sigma_t = _enforce_non_decreasing(_safe_positive(sigma_t))

    return MethodResult(
        tag="M2_QUADRATIC_LOCAL",
        E_target=float(max(E_t, 1e-12)),
        nu=el.nu,
        eps_target=eps_target,
        sigma_target=sigma_t,
    )


def compute_target_scaled_by_yield(el: ElasticData, pl: PlasticData, T_target: float, ref_rule: str | float) -> MethodResult:
    """
    M3: escala a curva de referência pelo ratio de σy(T_target)/σy(T_ref).
    Não gera "cauda" porque usa eps_ref da curva base.
    """
    E_t = piecewise_linear_in_T(el.temps, el.E, T_target)

    T_ref = _choose_ref_temperature(pl.temps, ref_rule, T_target)
    if T_ref not in pl.original_curves:
        T_ref = float(pl.temps[int(np.argmin(np.abs(pl.temps - T_ref)))])

    eps_ref, sig_ref = pl.original_curves[float(T_ref)]

    idx0 = int(np.argmin(np.abs(pl.eps_grid - 0.0)))
    sigma_y_T = pl.sigma_mat[:, idx0]
    sigma_y_target = piecewise_linear_in_T(pl.temps, sigma_y_T, T_target)

    idx0_ref = int(np.argmin(np.abs(eps_ref - 0.0)))
    sigma_y_ref = float(sig_ref[idx0_ref])
    if sigma_y_ref <= 0:
        raise ValueError("σy_ref <= 0 (dados de referência estranhos).")

    scale = float(sigma_y_target / sigma_y_ref)

    sig_t = sig_ref * scale
    sig_t = _enforce_non_decreasing(_safe_positive(sig_t))

    return MethodResult(
        tag=f"M3_SCALE_BY_YIELD_ref{T_ref:g}",
        E_target=float(max(E_t, 1e-12)),
        nu=el.nu,
        eps_target=np.asarray(eps_ref, dtype=float).copy(),
        sigma_target=np.asarray(sig_t, dtype=float).copy(),
    )


# =============================================================================
# 4) PLOTS (E(T) e curvas plásticas)
# =============================================================================

def plot_elastic(el: ElasticData, T_target: float, results: List[MethodResult],
                 export: bool, show: bool, filename: str):
    plt.figure()
    plt.plot(el.temps, el.E, marker="o", linestyle="-", label="Dados E(T)")
    for res in results:
        plt.scatter([T_target], [res.E_target], marker="x", label=f"{res.tag} @ {T_target:g}°C")
    plt.xlabel("Temperatura (°C)")
    plt.ylabel("E (unidades do input)")
    plt.title("Módulo elástico vs Temperatura")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if export:
        plt.savefig(filename, dpi=200)
    if show:
        plt.show()
    plt.close()


def plot_plastic(pl: PlasticData, T_target: float, results: List[MethodResult],
                 export: bool, show: bool, filename: str):
    plt.figure()

    for T in pl.temps:
        eps_i, sig_i = pl.original_curves[float(T)]
        plt.plot(eps_i, sig_i, linestyle="-", label=f"{T:g}°C")

    for res in results:
        plt.plot(res.eps_target, res.sigma_target, linestyle="--", linewidth=2,
                 label=f"{res.tag} @ {T_target:g}°C")

    plt.xlabel("Deformação plástica εp")
    plt.ylabel("Tensão σ (unidades do input)")
    plt.title("Curvas *PLASTIC + (interpolação/extrapolação) em T_TARGET")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if export:
        plt.savefig(filename, dpi=200)
    if show:
        plt.show()
    plt.close()


# =============================================================================
# 5) EXPORTAR CARTA COMPLETA ORDENADA POR TEMPERATURA
# =============================================================================

def _format_elastic_line(E: float, nu: float, T: float) -> str:
    return f"{E:>20.6f}, {nu:>12.6f}, {T:>12.6f}"


def _format_plastic_line(sigma: float, eps_p: float, T: float) -> str:
    return f"{sigma:>20.6f}, {eps_p:>16.9f}, {T:>12.6f}"


def export_full_material_inp(sections: MaterialSections,
                             el: ElasticData,
                             pl: PlasticData,
                             T_target: float,
                             res: MethodResult,
                             out_path: str):
    """
    Exporta um .inp contendo o MATERIAL completo:
    - preserva header, middle e footer como no input
    - reescreve *ELASTIC e *PLASTIC com as temperaturas em ordem crescente,
      inserindo T_target se não existir, ou substituindo o bloco de T_target se já existir
    """
    E_map = {float(T): float(E) for T, E in zip(el.temps, el.E)}
    E_map[float(T_target)] = float(res.E_target)

    temps_el = np.array(sorted(E_map.keys()), dtype=float)
    elastic_lines = [_format_elastic_line(E_map[float(T)], el.nu, float(T)) for T in temps_el]

    pl_map: Dict[float, Tuple[np.ndarray, np.ndarray]] = {float(T): pl.original_curves[float(T)] for T in pl.temps}
    pl_map[float(T_target)] = (res.eps_target, res.sigma_target)

    temps_pl = np.array(sorted(pl_map.keys()), dtype=float)

    plastic_lines: List[str] = []
    for T in temps_pl:
        eps_i, sig_i = pl_map[float(T)]
        sig_i = _enforce_non_decreasing(_safe_positive(np.asarray(sig_i, float)))
        eps_i = np.asarray(eps_i, float)
        for s, e in zip(sig_i, eps_i):
            plastic_lines.append(_format_plastic_line(float(s), float(e), float(T)))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("** ============================================================\n")
        f.write("** Auto-generated material card with interpolated/extrapolated T_TARGET\n")
        f.write(f"** T_TARGET = {T_target:g} °C | Method = {res.tag}\n")
        f.write(f"** EPS_DOMAIN_MODE={EPS_DOMAIN_MODE} | EPS_EXTRAP_MODE={EPS_EXTRAP_MODE}\n")
        f.write("** ============================================================\n\n")

        for l in sections.header_lines:
            f.write(l + "\n")

        f.write(sections.elastic_kw_line + "\n")
        for l in elastic_lines:
            f.write(l + "\n")

        for l in sections.middle_lines:
            f.write(l + "\n")

        f.write(sections.plastic_kw_line + "\n")
        for l in plastic_lines:
            f.write(l + "\n")

        for l in sections.footer_lines:
            f.write(l + "\n")


# =============================================================================
# 6) MAIN
# =============================================================================

def main():
    sections = split_material_sections(MATERIAL_TEXT)
    el = parse_elastic(sections.elastic_data_lines)
    pl = parse_plastic(sections.plastic_data_lines)

    results: List[MethodResult] = []

    r1 = compute_target_linear(el, pl, T_TARGET)
    r2 = compute_target_local_quadratic(el, pl, T_TARGET)
    r3 = compute_target_scaled_by_yield(el, pl, T_TARGET, SCALE_REF_T)

    results.extend([r1, r2, r3])

    figE = f"{OUTPUT_PREFIX}Elastic_E_vs_T_T{T_TARGET:g}.png"
    figP = f"{OUTPUT_PREFIX}Plastic_curves_T{T_TARGET:g}.png"

    plot_elastic(el, T_TARGET, results, EXPORT_FIGURES, SHOW_FIGURES, figE)
    plot_plastic(pl, T_TARGET, results, EXPORT_FIGURES, SHOW_FIGURES, figP)

    if EXPORT_CARDS:
        mat_name = "MATERIAL"
        m = re.search(r"\*MATERIAL\s*,\s*NAME\s*=\s*([^\s,]+)", MATERIAL_TEXT, flags=re.IGNORECASE)
        if m:
            mat_name = m.group(1).strip()

        for res in results:
            safe_tag = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", res.tag)
            out_inp = f"{OUTPUT_PREFIX}{mat_name}_T{T_TARGET:g}_{safe_tag}.inp"
            export_full_material_inp(sections, el, pl, T_TARGET, res, out_inp)

    print("OK.")
    if EXPORT_FIGURES:
        print(f"- Figuras: {figE} | {figP}")
    if EXPORT_CARDS:
        print("- Cartas .inp exportadas (uma por método), com sufixo do método no nome do arquivo.")


if __name__ == "__main__":
    main()
