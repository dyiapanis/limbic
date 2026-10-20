#!/usr/bin/env python3
"""Limbic recall eval harness — Stage A of the associative-linking scope.

Builds a fresh synthetic corpus, drives the REAL provider pipeline
(embedder + KNN + FTS + graph merge + scoring + confidence gate),
and reports recall on two surfaces:

  raw   — pre-gate ranking (provider._search, the R&D surface)
  gated — product surface (provider.recall, what the agent receives)

Deterministic: fixed corpus, fixed queries, real ONNX embedder.
Usage: eval/recall_eval.py [--home DIR]   (run with the gateway tools python)
Output: human table + one EVAL_JSON line for before/after diffing.
"""
import json
import os
import shutil
import sys
import tempfile
import time

UID = "eval-user"
K = 5

# 10 domains x 10 facts. Well-separated vocabulary per domain so
# cross-domain confusion is visible; paraphrased queries test semantic
# (not keyword) recall.
CORPUS = {
    "family": [
        "Demetri's mother Elena lives in Nicosia and calls every Sunday evening",
        "Elena is allergic to penicillin and carries a card for it",
        "Demetri's brother Andreas studied architecture in Athens",
        "Andreas and his wife Maya have twin daughters born in March",
        "The family gathers in Nicosia every Christmas for Elena's baking",
        "Demetri's father passed away three years ago; he was a carpenter",
        "Maya is a paediatric nurse at the general hospital",
        "Andreas restores vintage furniture as a weekend craft",
        "Elena grows roses and lemons on her small Nicosia terrace",
        "The twins are called Sofia and Lina and they are seven years old",
    ],
    "work": [
        "The flagship client project uses a Postgres-backed billing ledger",
        "Weekly sprint reviews happen every Tuesday at ten in the morning",
        "Priya leads the data engineering team of four analysts",
        "The staging server is named after moons of Jupiter",
        "Deployment freezes apply during the last week of every quarter",
        "The API gateway was rewritten in Rust for latency reasons",
        "Marcus handles vendor contracts and keeps the negotiation notes",
        "The team retro board lives in the shared workspace documents",
        "On-call rotation changes monthly and Priya schedules it",
        "The billing ledger discrepancy audit closed without findings",
    ],
    "health": [
        "Resting heart rate trends around fifty-eight on calm mornings",
        "The physio prescribed isometric holds for the left shoulder",
        "Vitamin D levels were low at the last blood panel in spring",
        "Sleep averages just under seven hours on weekdays",
        "The shoulder injury came from a fall while hiking in June",
        "Blood pressure readings sit comfortably in the normal band",
        "The dentist flagged one molar for a possible crown",
        "Zone-two runs are capped at thirty minutes while the shoulder heals",
        "Iron levels recovered after switching the supplement brand",
        "The physio follow-up is scheduled after the next scan",
    ],
    "travel": [
        "The Lisbon trip in May was booked through the airline directly",
        "The favourite Lisbon cafe served the custard tarts near the old square",
        "Backpacking through Vietnam happened the year after university",
        "The overnight train from Hanoi to Hue had sleeper berths",
        "Japan cherry-blossom season is best in the first April week",
        "The Vietnam motorbike loop through the northern hills took six days",
        "Airport lounge access comes with the travel credit card",
        "The Tokyo hostel was in Asakusa, two streets from the temple",
        "Passport expires next autumn and needs a renewal",
        "The Kyoto ryokan stay included a private onsen bath",
    ],
    "finance": [
        "The mortgage rate is fixed at three-point-one percent for ten years",
        "Index fund contributions happen on the first of each month",
        "The emergency fund covers roughly eight months of expenses",
        "The broker account charges no fee on ETF purchases",
        "Superannuation sits in the low-cost indexed option",
        "The electric bill averages about ninety in winter months",
        "The home insurance renews annually each September",
        "A small position in a robotics ETF was opened last year",
        "The car loan was paid off two years early to save interest",
        "Cashback on groceries goes to the travel fund automatically",
    ],
    "home": [
        "The hot water heater is a heat-pump unit in the garage",
        "The lemon tree by the deck needs deep watering weekly in summer",
        "The study desk is walnut and faces the north window",
        "The neighbours Maria and Paulo look after the cat when travelling",
        "The kitchen was repainted sage green during the last renovation",
        "Solar panels cover the western roof face",
        "The cat prefers the armchair by the bookshelf to the sofa",
        "The garage shelving was rebuilt with marine plywood",
        "A slow leak under the kitchen sink was fixed with a new washer",
        "The herb garden has rosemary, oregano and thyme in clay pots",
    ],
    "tech": [
        "The home server runs Debian with an unattended-upgrades policy",
        "The backup drive does versioned snapshots every night at two",
        "The e-reader runs a custom firmware with dark-mode fonts",
        "The mechanical keyboard uses silent tactile switches",
        "A tailnet connects the laptop, server and phone privately",
        "The photo library deduplication tool runs on the first Sunday",
        "The router firmware was replaced with the open variant",
        "Noise-cancelling headphones are charged by USB-C every evening",
        "The old laptop was repurposed as a kitchen recipe station",
        "Versioned dotfiles live in a private repository with a bootstrap script",
    ],
    "social": [
        "Theo hosts the monthly board-game evening with homemade pizza",
        "The walking group meets at the harbour fountain on Saturdays",
        "Nadia moved to Melbourne for a design role last year",
        "The board-game group favourite is a co-op about pandemic control",
        "The neighbour's farewell party was on the longest night of winter",
        "Nadia sends hand-illustrated postcards from every trip",
        "Theo is learning the accordion and practices at lunchtime",
        "The book club chose a maritime novel for the next meeting",
        "The harbour walks end with coffee at the corner bakery",
        "Reunions with the university crowd happen roughly every two years",
    ],
    "food": [
        "The sourdough starter is named Basil and lives in the fridge",
        "Weekly meal prep happens on Sunday with a big pot of grain and beans",
        "The best ramen in town is the tiny shop behind the station",
        "Basil the starter is five years old and fed with rye flour",
        "The spice drawer is organised by cuisine, not alphabetically",
        "The cast-iron pan is never washed with soap, only scraped and oiled",
        "The vegan cheese sauce recipe uses cashews and nutritional yeast",
        "Coffee is brewed with a pour-over at weekends, machine on weekdays",
        "The fermentation experiments last summer produced decent kimchi",
        "Fresh pasta night is a Friday tradition with the manual roller",
    ],
    "hobby": [
        "The film camera is a fully mechanical model from the seventies",
        "Darkroom printing happens in the converted bathroom cupboard",
        "The chess opening repertoire favours the Queen's Gambit",
        "The chess club ladder restarted at the bottom division",
        "Medium-format film gives twelve frames per roll",
        "The vinyl collection is sorted by label then by year",
        "Woodworking tools are honed on a water-stone before each project",
        "The current woodworking project is a dovetailed jewelry box",
        "Film development uses a daylight tank in the kitchen sink",
        "The record player needs a new belt after thirty years of service",
    ],
}

# 3 paraphrased queries per domain; expected = fact indices in that domain.
QUERIES = [
    ("family", "who in the family has a medicine allergy?", [1]),
    ("family", "where does the grandmother grow her fruit?", [6, 8]),
    ("family", "what do Andreas's kids do for school age?", [8]),
    ("work", "when do we hold the sprint review meeting?", [1]),
    ("work", "why was the gateway rewritten?", [5]),
    ("work", "who runs the on-call schedule?", [8, 2]),
    ("health", "what did the physio prescribe for the injury?", [1, 4]),
    ("health", "how is the sleep during the working week?", [3]),
    ("health", "any follow-ups after the blood panel?", [2, 9, 6]),
    ("travel", "which country trip involved motorbikes in hills?", [5, 3]),
    ("travel", "when should the travel document be renewed?", [8]),
    ("travel", "where in Tokyo did we sleep?", [7]),
    ("finance", "what are the housing-loan terms?", [0]),
    ("finance", "how big is the safety buffer?", [2]),
    ("finance", "what happens at the start of each month with investing?", [1]),
    ("home", "where does the cat like to nap?", [6]),
    ("home", "who waters the citrus in summer?", [1]),
    ("home", "what changed colour in the renovation?", [4]),
    ("tech", "how does the nightly versioned copy run?", [1]),
    ("tech", "what connects all the private devices?", [4]),
    ("tech", "what happened to the router software?", [6]),
    ("social", "who cooks for the games night?", [0]),
    ("social", "which friend moved states for work?", [2, 5]),
    ("social", "where do the weekend walks finish?", [8]),
    ("food", "what is the starter fed with?", [3]),
    ("food", "how is the iron pan cleaned?", [5]),
    ("food", "what happens on Sundays with the beans?", [1]),
    ("hobby", "what camera does the film photography use?", [0, 4]),
    ("hobby", "what is being built with the joinery tools?", [6, 7]),
    ("hobby", "how are the photographs developed at home?", [8, 1]),
]

# Contested mode: dense domains (15 facts each) with paraphrase queries whose
# expected facts must out-rank same-domain distractors — top-5 boundaries are
# genuinely contested here, unlike the well-separated base corpus.
CORPUS_CONTESTED = {
    "colleagues": [
        "Priya mentors the two graduate analysts and reviews their SQL every Friday",
        "Marcus keeps the vendor negotiation notes in a locked drawer and distrusts email",
        "Dana from the Melbourne office owns the analytics dashboard nobody else can edit",
        "Priya plays badminton on Thursday evenings with the graduate analysts",
        "Marcus brings his rescue greyhound to the office on slow Fridays",
        "Dana's dashboard migration to the new schema is blocked on her annual leave",
        "The graduate analysts wrote a style guide for SQL that Priya endorsed",
        "Marcus negotiated the printer lease down by a third last spring",
        "Dana runs the Melbourne stand-up at nine, an hour before ours",
        "Priya's team retro rule: no metrics, only stories, half an hour",
        "Marcus keeps a jar of liquorice sweets that everyone else pretends not to touch",
        "Dana's leave spans the last two weeks of the year every year",
        "The graduate analysts rotate the on-call pager on a fortnightly cycle",
        "Priya insisted the on-call schedule respect the analysts' coursework deadlines",
        "Marcus's greyhound is called Biscuit and sleeps under the reception desk",
    ],
    "recipes": [
        "The no-knead bread needs an overnight prove in the ceramic bowl with the lid",
        "Lentil bolognese gets its depth from soy sauce and a parmesan rind, not meat",
        "The overnight bread dough is folded once after twenty minutes, never kneaded",
        "Lentil bolognese was chosen because it freezes flat in bags for week lunches",
        "Granola bakes at a low temperature twice, with a stir in between, to keep clusters",
        "The granola recipe came from the corner bakery owner after a long chat",
        "Flat-freeze bags of sauce are labelled with the month in marker pen",
        "The pantry moth incident forced a full flour re-buy and sealed jars everywhere",
        "Every soup night starts from the same soffritto base in the big yellow pot",
        "The yellow pot was a wedding gift and never goes in the dishwasher",
        "Kimchi batches are started on the new moon because the ferment window suits weekends",
        "The pantry jars are labelled by grain, not by purchase date",
        "Soffritto night doubles as soup night and never coincides with bake night",
        "The bakery granola clusters survive only if the tray cools untouched",
        "A ferment jar lives on the top shelf, away from the stove heat",
    ],
    "trips": [
        "The overnight ferry to the island had reclining seats and a broken vending machine",
        "Island mornings were for the cliff path walk before the day-trippers arrived",
        "The ferry ticket was cheap because it left at five in the morning",
        "On the island we ate at the family taverna twice and never saw a menu",
        "The cliff path walk passes a ruined chapel with a lemon tree behind it",
        "The mountain hut trip involved carrying water because the spring was dry",
        "At the mountain hut we played cards by head-torch after the generator cut out",
        "The dry spring meant each person carried four litres up the scree",
        "The taverna family gave us a jar of their own thyme honey at the end",
        "Island taverna nights ended with the owner's rose liqueur, never a bill",
        "The ruined chapel lemon tree was heavy with fruit even that late in the season",
        "The generator at the hut ran only three hours each evening",
        "Cards by head-torch became the trip's longest-remembered joke about cheating",
        "The scree section took ninety minutes and two rest stops with the water weight",
        "Rose liqueur tastes better after the second thyme-honey spoonful, we decided",
    ],
    "devices": [
        "The smart thermostat learns the morning schedule and pre-heats by half an hour",
        "Thermostat pre-heating made the winter bill higher, not lower, the first month",
        "The robot vacuum gets tangled on the hallway rug every single run without fail",
        "The vacuum's cliff sensors were taped over once to clean the sunken lounge floor",
        "Taping the sensors voided something, probably, but the lounge floor got clean",
        "The hallway rug now has a rubber backing that seems to reduce the tangles",
        "The doorbell camera records only the top of visitors' heads because of its angle",
        "The doorbell was mounted before the new step was poured, hence the head-top angle",
        "The new front step is one brick higher than the old one",
        "The thermostat's learning can be reset by holding both dials for ten seconds",
        "The smart meter display lives face-down in a drawer because the pulsing light",
        "The meter face-down trick saves the bedroom wall from the blue pulse at night",
        "The vacuum dock is behind the sofa so it can't be seen from the kitchen",
        "The camera angle issue was reported to the installer who never came back",
        "Both dials for ten seconds also clears the thermostat's holiday mode",
    ],
}

QUERIES_CONTESTED = [
    ("colleagues", "who works with the junior data staff every week?", [0, 3, 6]),
    ("colleagues", "what is Marcus's dog situation at the office?", [4, 14]),
    ("colleagues", "how is Dana blocked on the schema work?", [5, 11]),
    ("colleagues", "what did the analysts write that got Priya's backing?", [6, 12]),
    ("recipes", "how is the bread dough handled before the overnight rise?", [0, 2]),
    ("recipes", "why do we make the lentil sauce in big batches?", [3, 6]),
    ("recipes", "what did the moth problem change in the kitchen storage?", [7, 11]),
    ("recipes", "what is the rule about the granola tray after baking?", [4, 13]),
    ("trips", "what was the morning routine on the island?", [1, 4]),
    ("trips", "how did the mountain water situation change the climb?", [6, 8, 13]),
    ("trips", "what happened at the hut when the power stopped?", [6, 11]),
    ("trips", "how did the taverna meals end?", [8, 9, 14]),
    ("devices", "why does the thermostat warm up early?", [0, 1]),
    ("devices", "what is the vacuum's recurring problem and the attempted fix?", [2, 4]),
    ("devices", "why does the doorbell only see heads?", [6, 7, 8]),
    ("devices", "how is the flashing meter handled at night?", [10, 11]),
]


def build_home() -> str:
    base = os.environ.get("TMPDIR") or tempfile.gettempdir()
    home = os.path.join(base, "limbic-eval-home")
    if os.path.exists(home):
        shutil.rmtree(home)
    os.makedirs(home)
    return home


def _arg(name: str, default: str) -> str:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main() -> int:
    home = _arg("--home", "") or build_home()
    passes = int(_arg("--passes", "2"))
    boost = float(_arg("--boost", "0.0"))
    corpus, queries = ((CORPUS_CONTESTED, QUERIES_CONTESTED)
                       if "--contested" in sys.argv else (CORPUS, QUERIES))
    if not os.path.isdir(home):
        os.makedirs(home, exist_ok=True)
    if boost > 0:
        # Feature config for this run only — scratch home, never the fleet's.
        with open(os.path.join(home, "limbic.yaml"), "w") as f:
            f.write(f"associative_link_boost: {boost}\n")
    print(f"eval home: {home}  passes={passes} boost={boost}")

    # The gateway runtime is isolated-mode: PYTHONPATH breaks its env
    # resolution. Insert paths in-process, exactly like the launcher does.
    sys.path.insert(0, os.environ.get("HERMES_AGENT_ROOT")
                   or os.path.expanduser("~/.hermes/hermes-agent"))
    # repo dir is itself the `limbic` package — put its PARENT on sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # eval/ -> repo parent, like pooling_ab.py
    from limbic.provider import LimbicMemoryProvider

    t0 = time.time()
    p = LimbicMemoryProvider()
    p.initialize("eval-session", hermes_home=home, user_id=UID, agent_identity="eval")
    # Repo's own test pattern: pin the user resolution — the scratch home has
    # no users.yaml manifest, and the harness must not depend on one.
    p._get_user_id = lambda sid: UID
    print(f"provider initialised ({time.time() - t0:.1f}s)")

    # Ingest — the REAL write path (embedder, trust, NLI bands, decay stamps).
    t0 = time.time()
    facts: list[tuple[str, str]] = []  # (content, domain)
    for domain, items in corpus.items():
        for content in items:
            res = p.remember(content, user_id=UID)
            if res.get("error"):
                print(f"STORE FAILED: {res['error']}: {content}", file=sys.stderr)
                return 1
            facts.append((content, domain))
    print(f"ingested {len(facts)} facts ({time.time() - t0:.1f}s)")

    # Query — both surfaces, N passes. Pass 1 exercises the real retrieval
    # chokepoint (prefetch: retrieval_count + edge formation); later passes
    # measure whether the links formed by earlier passes help recall.
    for pass_i in range(1, passes + 1):
        per_domain: dict[str, list[float]] = {}
        hit1 = 0
        abstained = 0
        gated_hits = 0
        gated_total = 0
        for domain, query, expected_idx in queries:
            expected_contents = {corpus[domain][i] for i in expected_idx}
            # prefetch = the real user flow: counts + (boost>0) link formation
            p.prefetch(query, session_id="eval-session")
            raw = p._search(query, UID, limit=K)
            got_contents = [r.get("content") for r in raw]
            raw_hits = sum(1 for c in got_contents if c in expected_contents)
            rec = raw_hits / len(expected_contents)
            per_domain.setdefault(domain, []).append(rec)
            if got_contents and got_contents[0] in expected_contents:
                hit1 += 1

            gated = p.recall(query, session_id="eval-session")
            if gated.get("status") == "no_confident_match":
                abstained += 1
            else:
                g_contents = [r["content"] for r in gated.get("results", [])]
                g_hits = sum(1 for c in g_contents if c in expected_contents)
                gated_hits += g_hits
                gated_total += len(expected_contents)

        raw_recall = sum(sum(v) / len(v) for v in per_domain.values()) / len(per_domain)
        macro = {d: round(sum(v) / len(v), 4) for d, v in per_domain.items()}
        result = {
            "pass": pass_i,
            "n_facts": len(facts),
            "n_queries": len(queries),
            "raw_recall@5": round(raw_recall, 4),
            "hit@1": round(hit1 / len(queries), 4),
            "gated_recall": round(gated_hits / gated_total, 4) if gated_total else 0.0,
            "abstained": abstained,
        }

        print(f"\n── pass {pass_i}/{passes} per-domain raw recall@5:")
        for d, v in sorted(macro.items()):
            print(f"  {d:<10} {v}")
        print(f"raw recall@5 (macro): {result['raw_recall@5']}")
        print(f"hit@1:                 {result['hit@1']}")
        print(f"gated recall:          {result['gated_recall']}  ({abstained} abstained)")
        print(f"EVAL_JSON={json.dumps(result)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())