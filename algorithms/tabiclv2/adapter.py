"""
Adapter SynthBridge per TabICLv2 (Qu, Holzmüller, Varoquaux, Le Morvan - ICML 2026).

Idea: TabICLv2 è un modello predittivo (classificazione e regressione). Per usarlo
come generatore si scompone la distribuzione congiunta con la regola della catena

    P(x1, ..., xd) = P(x1) * P(x2 | x1) * ... * P(xd | x1, ..., x(d-1))

e si campiona una colonna alla volta: ogni P(xk | x<k) è la distribuzione predittiva
di TabICLv2 (classificatore per le categoriche, 999 quantili per le numeriche),
ottenuta in-context usando le righe reali come "training set".
È quello che fa la classe ufficiale tabicl.TabICLUnsupervised; qui la usiamo
colonna per colonna per mostrare l'avanzamento nell'interfaccia.

Bilanciamento: con target_column + target_class si generano solo righe della classe
indicata, usando come contesto solo le righe reali di quella classe.
"""

import numpy as np
import pandas as pd


def generate(data: pd.DataFrame, n_rows: int, params: dict, info: dict) -> pd.DataFrame:
    from tabicl import TabICLUnsupervised

    # La barra di avanzamento è solo informativa: se l'aggiornamento fallisce
    # (es. su Windows un file bloccato da OneDrive o da un antivirus), si continua.
    _progress = info["progress"]

    def progress(fraction, message=""):
        try:
            _progress(fraction, message)
        except OSError as e:
            print(f"(avviso: barra di avanzamento non aggiornata: {e})")

    info = {**info, "progress": progress}

    seed = info.get("seed")
    seed = 42 if seed is None else int(seed)
    cat_cols = set(info["categorical_columns"])
    int_cols = set(info["integer_columns"])

    target = (params.get("target_column") or "").strip()
    target_class = (params.get("target_class") or "").strip()
    fixed = {}  # colonne con un valore costante nell'output

    # --- 1. Scelta del contesto (righe reali) e dell'ordine delle colonne -------
    df = data
    if target:
        if target not in data.columns:
            raise ValueError(f"La colonna target '{target}' non esiste. Colonne: {list(data.columns)}")
        if target_class:
            mask = (data[target].astype("string") == target_class).fillna(False).to_numpy()
            if mask.sum() == 0:
                valori = sorted(data[target].dropna().astype(str).unique().tolist())
                raise ValueError(f"La classe '{target_class}' non compare in '{target}'. Valori presenti: {valori}")
            df = data.loc[mask]
            fixed[target] = df[target].iloc[0]
            df = df.drop(columns=[target])
            print(f"Bilanciamento: genero solo righe con {target} = {target_class} "
                  f"(contesto: {len(df)} righe reali di quella classe).")
        else:
            # il target viene generato per primo, le altre colonne condizionate su di esso
            df = df[[target] + [c for c in data.columns if c != target]]

    max_ctx = int(params.get("max_context_rows", 3000))
    if len(df) > max_ctx:
        df = df.sample(n=max_ctx, random_state=seed)
        print(f"Contesto ridotto a {max_ctx} righe reali (sottocampione casuale).")
    if len(df) < 5:
        raise ValueError(f"Servono almeno 5 righe reali di contesto, ce ne sono {len(df)}.")

    # colonne costanti o tutte mancanti: non si modellano, si copiano
    model_cols = []
    for c in df.columns:
        valori = df[c].dropna()
        if valori.nunique() <= 1:
            fixed[c] = valori.iloc[0] if len(valori) else pd.NA
        else:
            model_cols.append(c)

    # --- 2. Codifica numerica: categoriche -> codici interi, mancanti -> NaN -----
    X = np.empty((len(df), len(model_cols)), dtype=np.float32)
    cat_idx, categories, ranges = [], {}, {}
    for j, c in enumerate(model_cols):
        if c in cat_cols:
            cat = pd.Categorical(df[c])
            codes = cat.codes.astype(np.float32)
            codes[codes < 0] = np.nan
            X[:, j] = codes
            cat_idx.append(j)
            categories[c] = cat.categories
        else:
            col = df[c].astype("Float64").to_numpy(dtype="float64", na_value=np.nan)
            X[:, j] = col
            ranges[c] = (np.nanmin(col), np.nanmax(col))

    # --- 3. Modello: TabICLv2 (pesi ufficiali scaricati al primo uso) ------------
    info["progress"](0.05, "caricamento di TabICLv2")
    device = params.get("device", "auto")
    model = TabICLUnsupervised(
        n_estimators=int(params.get("n_estimators", 4)),
        categorical_features=cat_idx,
        random_state=seed,
        device=None if device == "auto" else device,
    )
    model.fit(X)
    print(f"TabICLv2 pronto. Colonne modellate: {len(model_cols)} "
          f"({len(cat_idx)} categoriche), righe di contesto: {len(df)}.")

    # --- 4. Campionamento autoregressivo, una colonna alla volta -----------------
    temperature = float(params.get("temperature", 1.0))
    d = len(model_cols)
    if d > 0:
        rng = np.random.default_rng(seed)
        S = np.empty((n_rows, d), dtype=np.float32)
        for k, c in enumerate(model_cols):
            info["progress"](0.1 + 0.85 * k / d, f"colonna {k + 1}/{d}: {c}")
            train_mask = ~np.isnan(model.X_[:, k])
            if train_mask.sum() < 5:  # troppi mancanti: ricampiona i valori osservati
                valid = model.X_[train_mask, k]
                S[:, k] = rng.choice(valid, size=n_rows) if len(valid) else np.nan
                continue
            S[:, k] = model._sample_column(
                col_idx=k,
                cond_features=list(range(k)),
                X_test=S,
                train_mask=train_mask,
                temperature=temperature,
                rng=rng,
            )

    # --- 5. Decodifica: stesse colonne e tipi dei dati reali ---------------------
    info["progress"](0.97, "decodifica")
    out = pd.DataFrame(index=range(n_rows))
    for k, c in enumerate(model_cols):
        v = S[:, k].astype("float64")
        if c in categories:
            cats = categories[c]
            codes = np.clip(np.rint(v), 0, len(cats) - 1).astype(int)
            out[c] = pd.Series(np.asarray(cats, dtype=object)[codes])
        else:
            if params.get("clip_to_range", True):
                lo, hi = ranges[c]
                v = np.clip(v, lo, hi)
            if c in int_cols:
                out[c] = pd.Series(np.rint(v)).astype("Int64")
            else:
                out[c] = v
    for c, val in fixed.items():
        out[c] = [val] * n_rows

    return out[list(data.columns)]
