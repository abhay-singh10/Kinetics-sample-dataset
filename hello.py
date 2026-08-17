import csv
import time
from typing import Dict, List, Set

import yt_dlp


# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_CSV = "5000_hazard_youtube_links.csv"

TARGET_COUNT = 5000

# Each search returns up to 150 candidates.
SEARCH_LIMIT = 150

# Only retain videos between 3 and 180 seconds.
MIN_DURATION = 3
MAX_DURATION = 180


# =============================================================================
# STRONG TITLE BLACKLIST
# =============================================================================

TITLE_BLACKLIST = {
    # -------------------------------------------------------------------------
    # Medical / educational / commentary
    # -------------------------------------------------------------------------
    "doctor",
    "medical",
    "medicine",
    "symptoms",
    "symptom",
    "treatment",
    "therapy",
    "medication",
    "cure",
    "diagnosis",
    "diagnostic",
    "epilepsy explained",
    "seizure explained",
    "explaining",
    "explained",
    "discussion",
    "interview",
    "podcast",
    "talk show",
    "lecture",
    "lesson",
    "class",
    "course",
    "education",
    "educational",
    "history of",
    "documentary",
    "news",
    "news report",
    "report",
    "commentary",
    "analysis",

    # -------------------------------------------------------------------------
    # YouTube reaction / vlog content
    # -------------------------------------------------------------------------
    "reaction",
    "reacting",
    "reacts",
    "vlog",
    "daily vlog",
    "review",
    "reviewing",
    "unboxing",
    "testing my",
    "my setup",
    "behind the scenes",

    # -------------------------------------------------------------------------
    # Video editing / software tutorials
    # -------------------------------------------------------------------------
    "tutorial",
    "how to",
    "how i",
    "how do",
    "guide",
    "step by step",
    "for beginners",
    "beginner",
    "learn",
    "tips and tricks",
    "premiere",
    "premiere pro",
    "after effects",
    "aftereffects",
    "ae tutorial",
    "capcut",
    "davinci",
    "davinci resolve",
    "resolve",
    "final cut",
    "final cut pro",
    "alight motion",
    "photoshop",
    "preset",
    "presets",
    "plugin",
    "plugins",
    "template",
    "templates",
    "transition",
    "transitions",
    "overlay pack",
    "editing tutorial",
    "video editing",
    "greenscreen",
    "green screen",
    "lut",
    "keyframes",
    "rendering",
    "render",
    "effect tutorial",
    "edit tutorial",

    # -------------------------------------------------------------------------
    # DIY / electronics / hardware
    # -------------------------------------------------------------------------
    "diy",
    "how to make",
    "how to build",
    "how to create",
    "arduino",
    "esp32",
    "esp8266",
    "raspberry pi",
    "555 timer",
    "555",
    "circuit",
    "circuits",
    "breadboard",
    "schematic",
    "pcb",
    "soldering",
    "wiring",
    "resistor",
    "transistor",
    "electronics",
    "electronic",
    "homemade",
    "homemade strobe",
    "led strip build",
    "led build",
    "hardware build",
    "engineering",

    # -------------------------------------------------------------------------
    # Photography / lighting equipment
    # -------------------------------------------------------------------------
    "godox",
    "neewer",
    "profoto",
    "speedlight",
    "flashgun",
    "softbox",
    "studio strobe",
    "lighting setup",
    "lighting gear",
    "portrait photography",
    "camera flash",
    "camera lighting",
    "aputure",
    "trigger",
    "diffuser",
    "umbrella",
    "flash photography",

    # -------------------------------------------------------------------------
    # Relaxation / audio / non-visual content
    # -------------------------------------------------------------------------
    "relaxation",
    "relax",
    "sleep",
    "sleep hypnosis",
    "meditation",
    "guided",
    "hypnosis",
    "asmr",
    "audiobook",
    "soundtrack",
    "full album",
    "album",
    "lofi",
    "chill",
    "study music",
    "sleep music",
    "white noise",

    # -------------------------------------------------------------------------
    # Magic / drawing / art
    # -------------------------------------------------------------------------
    "drawing",
    "draw",
    "painting",
    "paint",
    "sketching",
    "sketch",
    "art tutorial",
    "magic trick",
    "magic reveal",
    "illusion explained",

    # -------------------------------------------------------------------------
    # Gaming / long playthrough content
    # -------------------------------------------------------------------------
    "gameplay",
    "playthrough",
    "longplay",
    "walkthrough",
    "game review",
    "gaming",

    # -------------------------------------------------------------------------
    # Music / performance terms likely to produce irrelevant videos
    # -------------------------------------------------------------------------
    "official music video",
    "music video",
    "lyrics",
    "lyric video",
    "song",
    "track",
    "dj mix",
    "mix",
    "full concert",
    "concert vlog",
}


# =============================================================================
# SEARCH QUERIES
# =============================================================================

QUERIES = [

    # -------------------------------------------------------------------------
    # STROBE / FLICKER / FLASH
    # -------------------------------------------------------------------------

    'epilepsy warning flashing screen -doctor -podcast -symptoms -treatment -documentary -vlog',
    'seizure warning strobe light -medical -cure -explaining -review -interview',
    'photosensitivity warning screen flicker -doctor -diagnosis -reaction',
    'flash warning strobe effects -tutorial -commentary -vlog',
    'strobe light visual effect screen test -unboxing -review -setup',
    'extreme strobe light test screen -unboxing -lighting rig',
    'high frequency flashing screen visual test -tutorial',
    'rapid screen flicker test visual hazard -talk',
    'concert stage strobe visualizer loop -full set -crowd vlog',
    'nightclub strobe light effects visualizer loop -dj mix -live stream',
    'lightning storm rapid flashes compilation -documentary -news',
    'arcade game intense screen flash loop -playthrough -longplay',
    'vj loop strobe flash abstract visualizer -tutorial',
    'hardcore flashing lights visual loop -music track -album',
    'rapid flash visual test screen loop -tutorial -diy',
    'flashing screen seizure warning loop -premiere -capcut -tutorial',
    'epilepsy warning strobe loop -tutorial -diy -editing',
    'strobe visualizer loop flashing screen -tutorial -how',
    'rapid screen flicker visual test -circuit -arduino -tutorial',
    '10hz strobe test screen -tutorial -diy -circuit -arduino -build',
    '15hz flicker visual test screen -circuit -arduino -how -tutorial',
    '20hz strobe flashing screen test -tutorial -diy',
    'extreme strobe light screen test -unboxing -lighting -review',
    'black and white strobe screen loop -tutorial -diy -camera',
    'vj strobe loop background -preset -pack -tutorial -template',
    'saturated red flash screen test -tutorial -how -code',
    'red blue strobe flashing screen loop -police -chase -news -diy',
    'photosensitivity test flashing screen -doctor -medical -treatment -symptoms',

    # -------------------------------------------------------------------------
    # COLOR / CHROMATIC FLASHES
    # -------------------------------------------------------------------------

    'saturated red flash visual test screen -tutorial',
    'red blue strobe light flashing loop -siren -asmr',
    'chromatic color flicker visual hazard test -review',
    'emergency strobe light flash animation loop -police chase -news',
    'pure red flashing screen seizure warning -doctor -medical',
    'rgb color swap rapid strobe loop -coding -tutorial',
    'neon strobe flashing sequence visual loop -diy -how',
    'cyberpunk flashing neon strobe visualizer -radio -podcast',
    'laser show rapid strobe visual effects -full concert -vlog',
    'police strobe flash sequence test -unboxing -review',

    # -------------------------------------------------------------------------
    # SPIRALS / HYPNOTIC / MOTION ILLUSIONS
    # -------------------------------------------------------------------------

    'spinning fraser spiral optical illusion loop -drawing -tutorial',
    'hypnotic spiral rotating animation loop -music -song -sleep -therapy',
    'dizzy spinning optical illusion video loop -magic trick -tutorial',
    'hypnotic black and white spiral vortex loop -meditation -relaxation',
    'fraser spiral motion illusion test -explained -history',
    'rotating op art visual illusion motion loop -drawing -painting',
    'vertigo visual distortion motion spiral loop -treatment -cure',
    'hypnotic swirl spinning visual disorientation -voiceover -guided',
    'endless spinning spiral motion illusion -asmr -audiobook',
    'motion aftereffect spiral illusion waterfall loop -lesson -class',
    'rotating spiral visual illusion loop',
    'fast spinning spiral optical illusion',
    'spiral tunnel motion illusion loop',
    'black white rotating spiral illusion',
    'rapid rotating pattern visual illusion',

    # -------------------------------------------------------------------------
    # GRIDS / GEOMETRY / PATTERN GLARE
    # -------------------------------------------------------------------------

    'scintillating grid optical illusion moving animation -explained',
    'hermann grid illusion moving animation loop -lesson',
    'hypnotic moving tunnel animation loop -sleep music -study',
    'pulsating concentric rings optical illusion loop -drawing -relax',
    'pattern glare visual stress test grating motion -doctor -symptoms',
    'high contrast stripe grating moving illusion -exam -treatment',
    'infinite checkerboard tunnel optical illusion loop -gameplay',
    'psychedelic optical illusion tunnel moving loop -audiobook',
    'warp speed optical tunnel visualizer loop -space documentary',
    'visual stress test moving grating lines -optometry -clinic',
    'moving high contrast grid visual illusion loop',
    'pulsating checkerboard visual illusion',
    'rapid moving stripe pattern visual loop',
    'concentric circles pulsing illusion loop',
    'high contrast grating motion visual test',

    # -------------------------------------------------------------------------
    # SHORTS
    # -------------------------------------------------------------------------

    '#shorts #epilepsywarning -doctor -podcast -symptoms',
    '#shorts #seizurewarning -treatment -cure',
    '#shorts #strobelight -unboxing -review',
    '#shorts #flashinglights -vlog -reaction',
    '#shorts #opticalillusion -drawing -tutorial -magic',
    '#shorts #hypnoticspiral -sleep -meditation',
    '#shorts #hypnotic -guided -hypnosis session',
    '#shorts #patternglare -symptoms -clinic',
    '#shorts #visualstress -diagnosis -optometrist',
    '#shorts #trippyvisuals -music video -song',
    '#shorts #mindbendingillusion -magic trick -reveal',
    '#shorts flashing screen strobe loop',
    '#shorts rapid flicker visual test',
    '#shorts spinning optical illusion',
    '#shorts moving pattern illusion',
]


# =============================================================================
# HELPERS
# =============================================================================

def normalize_title(title: str) -> str:
    return " ".join((title or "").lower().split())


def title_is_valid(title: str) -> bool:
    """
    Reject titles that strongly indicate tutorials, commentary,
    equipment, medical content, music/audio, etc.
    """
    title_lower = normalize_title(title)

    for keyword in TITLE_BLACKLIST:
        if keyword in title_lower:
            return False

    return True


def run_search(query: str) -> List[Dict]:
    """
    Search YouTube without downloading video bytes.
    flat-playlist gives us lightweight candidate entries.
    """

    search_url = f"ytsearch{SEARCH_LIMIT}:{query}"

    opts = {
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "skip_download": True,
        "extract_flat": True,
    }

    candidates = []

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            result = ydl.extract_info(search_url, download=False)

        if not result:
            return candidates

        for entry in result.get("entries", []) or []:
            if not entry:
                continue

            video_id = entry.get("id")
            title = entry.get("title", "")

            if not video_id:
                continue

            # Search results can sometimes contain non-video entries.
            # We only retain YouTube video IDs.
            if len(video_id) != 11:
                continue

            candidates.append({
                "video_id": video_id,
                "title": title,
            })

    except Exception as e:
        print(f"    Search error: {e}")

    return candidates


def get_video_metadata(video_id: str):
    """
    Fetch actual video metadata for duration/live validation.
    This does NOT download the video.
    """

    url = f"https://www.youtube.com/watch?v={video_id}"

    opts = {
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "skip_download": True,
    }

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)

        if not info:
            return None

        return info

    except Exception:
        return None


def is_valid_metadata(info: Dict) -> bool:
    """
    Final metadata-level filtering.
    """

    if not info:
        return False

    # Reject livestreams / upcoming / live-like content.
    if info.get("is_live") is True:
        return False

    live_status = info.get("live_status")

    if live_status in {"is_live", "is_upcoming"}:
        return False

    duration = info.get("duration")

    if duration is None:
        return False

    if duration < MIN_DURATION:
        return False

    if duration > MAX_DURATION:
        return False

    title = info.get("title", "")

    if not title_is_valid(title):
        return False

    return True


# =============================================================================
# MAIN EXTRACTION
# =============================================================================

def run_extraction():

    records = []

    seen_ids: Set[str] = set()

    candidate_ids: Set[str] = set()

    print("=" * 80)
    print("YOUTUBE HAZARD VIDEO DATASET EXTRACTION")
    print("=" * 80)

    print(f"Target unique videos : {TARGET_COUNT}")
    print(f"Search queries       : {len(QUERIES)}")
    print(f"Results/query        : {SEARCH_LIMIT}")
    print(f"Duration             : {MIN_DURATION}s - {MAX_DURATION}s")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 1: Search all queries and create a candidate pool
    # -------------------------------------------------------------------------

    for i, query in enumerate(QUERIES, start=1):

        if len(candidate_ids) >= TARGET_COUNT * 2:
            print("\nCandidate pool is sufficiently large. Stopping searches.")
            break

        print(
            f"\n[{i:02d}/{len(QUERIES)}] "
            f"SEARCH: {query[:90]}"
        )

        candidates = run_search(query)

        added = 0

        for candidate in candidates:

            video_id = candidate["video_id"]
            title = candidate["title"]

            if video_id in candidate_ids:
                continue

            # Apply cheap title filter first.
            if not title_is_valid(title):
                continue

            candidate_ids.add(video_id)
            added += 1

        print(
            f"    Search results: {len(candidates)}"
        )

        print(
            f"    New candidates: {added}"
        )

        print(
            f"    Candidate pool : {len(candidate_ids)}"
        )

    # -------------------------------------------------------------------------
    # STEP 2: Fetch metadata and perform accurate validation
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("VALIDATING CANDIDATES")
    print("=" * 80)

    total_candidates = len(candidate_ids)

    for index, video_id in enumerate(candidate_ids, start=1):

        if len(records) >= TARGET_COUNT:
            break

        if index % 25 == 0 or index == 1:
            print(
                f"[{index}/{total_candidates}] "
                f"Accepted: {len(records)}/{TARGET_COUNT}"
            )

        info = get_video_metadata(video_id)

        if not is_valid_metadata(info):
            continue

        title = info.get("title", "").strip()
        duration = int(round(info.get("duration", 0)))

        webpage_url = info.get(
            "webpage_url",
            f"https://www.youtube.com/watch?v={video_id}"
        )

        matched_query = info.get(
            "search_query",
            ""
        )

        records.append({
            "video_id": video_id,
            "title": title,
            "duration_sec": duration,
            "url": webpage_url,
            "matched_query": matched_query,
        })

        print(
            f"    + {len(records):04d} | "
            f"{duration:3d}s | {title[:80]}"
        )

    # -------------------------------------------------------------------------
    # STEP 3: Save CSV
    # -------------------------------------------------------------------------

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_id",
                "title",
                "duration_sec",
                "url",
                "matched_query",
            ]
        )

        writer.writeheader()
        writer.writerows(records)

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)

    print(f"Unique valid videos : {len(records)}")
    print(f"Target              : {TARGET_COUNT}")
    print(f"Candidate pool      : {len(candidate_ids)}")
    print(f"CSV                 : {OUTPUT_CSV}")

    if len(records) < TARGET_COUNT:
        print(
            "\nWARNING:"
            "\nOnly "
            f"{len(records)} "
            "valid videos were found."
            "\nRun again with additional search queries if you need "
            "more than this."
        )

    print("=" * 80)


if __name__ == "__main__":
    run_extraction()