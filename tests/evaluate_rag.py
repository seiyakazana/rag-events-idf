#!/usr/bin/env python3
"""
evaluate_rag.py
---------------
Évalue le pipeline RAG sur le jeu de test annoté tests/qa_test_dataset.json.

Modes d'évaluation
  keyword  — vérifie la présence/absence de mots-clés dans la réponse (rapide, sans surcoût API)
  llm      — LLM-as-judge via Mistral (évaluation sémantique, consomme des tokens)
  both     — les deux (défaut)

Usage
  python tests/evaluate_rag.py
  python tests/evaluate_rag.py --mode keyword
  python tests/evaluate_rag.py --ids tc_001,tc_005,tc_018
  python tests/evaluate_rag.py --verbose --output tests/eval_results/my_run.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

load_dotenv(ROOT_DIR / ".env")

# ── Constants ─────────────────────────────────────────────────────────────────

DATASET_PATH    = ROOT_DIR / "tests" / "qa_test_dataset.json"
RESULTS_DIR     = ROOT_DIR / "tests" / "eval_results"
JUDGE_MODEL     = os.environ.get("JUDGE_MODEL", "mistral-small-latest")
RAG_DELAY_S     = float(os.environ.get("RAG_EVAL_DELAY", 1.5))   # pause between RAG calls

# Mots-clés qui signalent qu'une réponse indique l'absence de résultat
NO_RESULT_SIGNALS = [
    "aucun événement",
    "aucun résultat",
    "pas d'événement",
    "ne correspond pas",
    "ne trouve pas",
    "ne dispose pas",
    "je ne trouve",
    "introuvable",
    "hors de mon",
    "hors du périmètre",
]

# ── Scoring : mots-clés ───────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    """Minuscules + suppression des accents pour comparaison souple."""
    return unicodedata.normalize("NFD", s.lower()).encode("ascii", "ignore").decode()


def _contains(text: str, keyword: str) -> bool:
    """Recherche insensible à la casse et aux accents."""
    return _normalize(keyword) in _normalize(text)


def score_keywords(response: str, case: dict) -> dict[str, Any]:
    """
    Évalue la réponse par présence/absence de mots-clés.

    Retourne un dict avec :
      keyword_score   int  0-100
      must_hits       list[str]  mots-clés must_mention trouvés
      must_misses     list[str]  mots-clés must_mention absents
      forbidden_hits  list[str]  mots-clés must_not_mention trouvés (violations)
      no_result_ok    bool|None  vérification du type "no_result" si applicable
    """
    elem = case.get("expected_elements", {})
    must_mention     = elem.get("must_mention", [])
    must_not_mention = elem.get("must_not_mention", [])
    answer_type      = case.get("answer_type", "")

    must_hits    = [kw for kw in must_mention     if _contains(response, kw)]
    must_misses  = [kw for kw in must_mention     if not _contains(response, kw)]
    forbidden    = [kw for kw in must_not_mention if _contains(response, kw)]

    # Score de base : % de must_mention trouvés
    base = (len(must_hits) / len(must_mention) * 100) if must_mention else 100.0

    # Pénalité : chaque violation must_not_mention retire 20 points
    penalty = len(forbidden) * 20
    keyword_score = max(0, int(base - penalty))

    # Vérification spécifique no_result
    no_result_ok = None
    if answer_type == "no_result":
        no_result_ok = any(_contains(response, sig) for sig in NO_RESULT_SIGNALS)
        if not no_result_ok:
            keyword_score = max(0, keyword_score - 30)

    return {
        "keyword_score":  keyword_score,
        "must_hits":      must_hits,
        "must_misses":    must_misses,
        "forbidden_hits": forbidden,
        "no_result_ok":   no_result_ok,
    }


def keyword_pass(kw_result: dict, case: dict) -> bool:
    """Seuil de passage : score >= 60 et aucune violation must_not_mention."""
    if kw_result["forbidden_hits"]:
        return False
    if case.get("answer_type") == "no_result" and kw_result["no_result_ok"] is False:
        return False
    return kw_result["keyword_score"] >= 60


# ── Scoring : LLM-as-judge ────────────────────────────────────────────────────

JUDGE_SYSTEM = """\
Tu es un évaluateur expert de systèmes RAG (Retrieval-Augmented Generation).
Tu reçois une question posée à un assistant événementiel Île-de-France, \
la réponse de l'assistant, et des critères d'évaluation.
Tu dois noter la réponse de 1 à 5 selon ces critères et donner une courte justification.

Barème :
1 = Réponse hors sujet, incorrecte ou hallucinations graves
2 = Partiellement pertinente, erreurs importantes
3 = Correcte mais incomplète ou imprécise
4 = Bonne réponse, légères lacunes
5 = Réponse parfaite selon les critères

Réponds UNIQUEMENT avec ce format JSON (rien d'autre) :
{"score": <int 1-5>, "reason": "<explication courte en français>"}
"""


def score_with_llm(
    question: str,
    response: str,
    case: dict,
    client,
) -> dict[str, Any]:
    """Évalue la réponse via LLM-as-judge Mistral. Retourne score 1-5 et raison."""
    criteria = case.get("evaluation_criteria", "Évalue la pertinence et l'exactitude.")

    user_msg = (
        f"**Question posée :**\n{question}\n\n"
        f"**Réponse de l'assistant :**\n{response}\n\n"
        f"**Critères d'évaluation :**\n{criteria}"
    )

    try:
        chat_resp = client.chat.complete(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = chat_resp.choices[0].message.content.strip()
        # Extraire le JSON même si le modèle ajoute du texte autour
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return {
                "llm_score":  int(data.get("score", 0)),
                "llm_reason": data.get("reason", ""),
                "llm_raw":    raw,
                "llm_error":  None,
            }
        return {"llm_score": 0, "llm_reason": "", "llm_raw": raw, "llm_error": "JSON non trouvé dans la réponse du juge"}
    except Exception as exc:
        return {"llm_score": 0, "llm_reason": "", "llm_raw": "", "llm_error": str(exc)}


def llm_pass(llm_result: dict) -> bool:
    return llm_result.get("llm_score", 0) >= 3


# ── Évaluation principale ─────────────────────────────────────────────────────

def run_evaluation(
    cases: list[dict],
    chain,
    client,
    mode: str,
    verbose: bool,
) -> list[dict]:
    results = []

    for i, case in enumerate(cases, 1):
        tc_id    = case["id"]
        question = case["question"]

        print(f"\n[{i:02d}/{len(cases)}] {tc_id} ({case['category']}, {case['difficulty']})")
        print(f"  Q: {question[:90]}{'…' if len(question) > 90 else ''}")

        # ── Appel RAG ────────────────────────────────────────────────────────
        t0 = time.time()
        try:
            response = chain.invoke(question)
            latency  = round(time.time() - t0, 2)
            rag_error = None
        except Exception as exc:
            response  = ""
            latency   = round(time.time() - t0, 2)
            rag_error = str(exc)
            print(f"  !! Erreur RAG : {rag_error}")

        if verbose:
            print(f"  R: {response[:300]}{'…' if len(response) > 300 else ''}")

        result: dict[str, Any] = {
            "id":         tc_id,
            "category":   case["category"],
            "difficulty": case["difficulty"],
            "question":   question,
            "response":   response,
            "latency_s":  latency,
            "rag_error":  rag_error,
        }

        # ── Scoring mots-clés ────────────────────────────────────────────────
        if mode in ("keyword", "both"):
            kw = score_keywords(response, case)
            result.update(kw)
            result["keyword_pass"] = keyword_pass(kw, case)
            status = "[OK]" if result["keyword_pass"] else "[KO]"
            print(f"  Keyword score : {kw['keyword_score']:3d}/100  {status}")
            if kw["must_misses"]:
                print(f"    Absents      : {kw['must_misses']}")
            if kw["forbidden_hits"]:
                print(f"    Violations   : {kw['forbidden_hits']}")

        # ── Scoring LLM-as-judge ─────────────────────────────────────────────
        if mode in ("llm", "both") and not rag_error:
            time.sleep(RAG_DELAY_S)   # éviter le rate-limit
            llm = score_with_llm(question, response, case, client)
            result.update(llm)
            result["llm_pass"] = llm_pass(llm)
            if llm["llm_error"]:
                print(f"  LLM judge err : {llm['llm_error']}")
            else:
                status = "[OK]" if result["llm_pass"] else "[KO]"
                print(f"  LLM score     : {llm['llm_score']}/5  {status}  -- {llm['llm_reason'][:80]}")

        results.append(result)

        # Pause entre les appels RAG pour ne pas saturer l'API
        if i < len(cases):
            time.sleep(RAG_DELAY_S)

    return results


# ── Affichage du résumé ───────────────────────────────────────────────────────

def print_summary(results: list[dict], mode: str) -> None:
    total = len(results)
    print("\n" + "=" * 70)
    print("RÉSUMÉ DE L'ÉVALUATION")
    print("=" * 70)

    if mode in ("keyword", "both"):
        scores = [r["keyword_score"] for r in results if "keyword_score" in r]
        passes = [r for r in results if r.get("keyword_pass")]
        print(f"\nKeyword scoring ({len(scores)} cas)")
        print(f"  Score moyen    : {sum(scores)/len(scores):.1f}/100" if scores else "  —")
        print(f"  Taux de succès : {len(passes)}/{total}  ({100*len(passes)//total}%)")

        # Par catégorie
        categories = sorted({r["category"] for r in results})
        print("\n  Par catégorie :")
        for cat in categories:
            cat_res = [r for r in results if r["category"] == cat and "keyword_score" in r]
            if not cat_res:
                continue
            avg = sum(r["keyword_score"] for r in cat_res) / len(cat_res)
            ok  = sum(1 for r in cat_res if r.get("keyword_pass"))
            print(f"    {cat:<25} {ok}/{len(cat_res)}  avg={avg:.0f}/100")

        # Par difficulté
        print("\n  Par difficulté :")
        for diff in ("facile", "moyen", "difficile"):
            diff_res = [r for r in results if r["difficulty"] == diff and "keyword_score" in r]
            if not diff_res:
                continue
            avg = sum(r["keyword_score"] for r in diff_res) / len(diff_res)
            ok  = sum(1 for r in diff_res if r.get("keyword_pass"))
            print(f"    {diff:<12} {ok}/{len(diff_res)}  avg={avg:.0f}/100")

    if mode in ("llm", "both"):
        llm_res = [r for r in results if r.get("llm_score", 0) > 0]
        if llm_res:
            avg_llm  = sum(r["llm_score"] for r in llm_res) / len(llm_res)
            llm_pass_count = sum(1 for r in llm_res if r.get("llm_pass"))
            print(f"\nLLM-as-judge scoring ({len(llm_res)} cas évalués)")
            print(f"  Score moyen    : {avg_llm:.2f}/5")
            print(f"  Taux de succès : {llm_pass_count}/{len(llm_res)}  ({100*llm_pass_count//len(llm_res)}%)")

    # Tableau des échecs
    failures = [
        r for r in results
        if not r.get("keyword_pass", True) or r.get("rag_error")
    ]
    if failures:
        print("\n  Cas en échec :")
        for r in failures:
            ks = r.get("keyword_score", "N/A")
            ls = r.get("llm_score", "—")
            print(f"    {r['id']}  keyword={ks}/100  llm={ls}/5  {r.get('rag_error') or ''}")

    avg_latency = sum(r["latency_s"] for r in results) / total
    print(f"\nLatence RAG moyenne : {avg_latency:.2f}s")
    print("=" * 70)


# ── Sauvegarde ────────────────────────────────────────────────────────────────

def save_results(results: list[dict], output_path: Path, dataset_meta: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now().isoformat(),
        "dataset_version": dataset_meta.get("version", "unknown"),
        "total_cases": len(results),
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nRapport sauvegardé : {output_path}")


# ── Point d'entrée ────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Évaluation du pipeline RAG Île-de-France")
    p.add_argument("--dataset", default=str(DATASET_PATH),
                   help="Chemin vers qa_test_dataset.json")
    p.add_argument("--mode", choices=["keyword", "llm", "both"], default="both",
                   help="Mode d'évaluation (défaut : both)")
    p.add_argument("--ids", default="",
                   help="IDs de cas à exécuter, séparés par des virgules (ex: tc_001,tc_005)")
    p.add_argument("--output", default="",
                   help="Chemin du fichier de résultats JSON (défaut : tests/eval_results/eval_TIMESTAMP.json)")
    p.add_argument("--verbose", action="store_true",
                   help="Afficher les réponses complètes du RAG")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Chargement du dataset ──────────────────────────────────────────────
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Erreur : dataset introuvable → {dataset_path}", file=sys.stderr)
        sys.exit(1)

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    cases = dataset["test_cases"]

    if args.ids:
        requested = set(args.ids.split(","))
        cases = [c for c in cases if c["id"] in requested]
        if not cases:
            print(f"Aucun cas trouvé pour : {args.ids}", file=sys.stderr)
            sys.exit(1)

    print(f"Jeu de test     : {dataset_path.name}  (v{dataset['metadata']['version']})")
    print(f"Cas sélectionnés: {len(cases)}/{len(dataset['test_cases'])}")
    print(f"Mode            : {args.mode}")

    # ── Chargement du pipeline RAG ─────────────────────────────────────────
    print("\nChargement du pipeline RAG…")
    try:
        from app.rag import build_rag_chain
        chain = build_rag_chain()
    except Exception as exc:
        print(f"Erreur lors du chargement du RAG : {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Client Mistral pour le juge LLM ───────────────────────────────────
    client = None
    if args.mode in ("llm", "both"):
        try:
            from mistralai.client import Mistral
            api_key = os.environ.get("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("MISTRAL_API_KEY manquante dans .env")
            client = Mistral(api_key=api_key)
            print(f"Juge LLM        : {JUDGE_MODEL}")
        except Exception as exc:
            print(f"Avertissement : LLM judge désactivé ({exc}). Passage en mode 'keyword'.")
            args.mode = "keyword"

    # ── Exécution ──────────────────────────────────────────────────────────
    print(f"\nDébut de l'évaluation — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results = run_evaluation(cases, chain, client, args.mode, args.verbose)

    # ── Résumé ─────────────────────────────────────────────────────────────
    print_summary(results, args.mode)

    # ── Sauvegarde ─────────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        out_path  = RESULTS_DIR / f"eval_{timestamp}.json"

    save_results(results, out_path, dataset["metadata"])


if __name__ == "__main__":
    main()
