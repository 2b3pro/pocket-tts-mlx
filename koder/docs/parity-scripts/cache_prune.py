"""Find (and with --delete, remove) pocket-tts cache files the port at HEAD never loads."""
import os, sys
from pathlib import Path
HUB = Path.home() / ".cache/huggingface/hub"
REPO = HUB / "models--kyutai--pocket-tts-without-voice-cloning"
VOICE_REV, LEGACY_REV, KEEP_REVS = "1e08e6a23401048648a9fdcfde2f89348215c2a7", "d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3", {"00eac05ed3d16bdc3f6b5d598874019c34a89214", "2578fed2380333b621689eaed6fe144cf69dfeb3"}
needed = set(Path(sys.argv[1]).read_text().split())
stale = []
for link in sorted((REPO / "snapshots").rglob("*")):
    if not link.is_symlink():
        continue
    rev, rel = link.relative_to(REPO / "snapshots").parts[0], "/".join(link.relative_to(REPO / "snapshots").parts[1:])
    keep = f"{rel}@{rev}" in needed or rev in KEEP_REVS or rev == LEGACY_REV or (rev == VOICE_REV and "/embeddings/" in rel)
    if not keep:
        stale.append(link)
def resolve_chain(link):
    repo_blob = (link.parent / os.readlink(link)).resolve(strict=False) if False else Path(os.path.normpath(link.parent / os.readlink(link)))
    shared = Path(os.path.normpath(repo_blob.parent / os.readlink(repo_blob))) if repo_blob.is_symlink() else None
    return repo_blob, shared
all_links = [p for p in HUB.rglob("*") if p.is_symlink()]
def referrers(target, exclude):
    return [p for p in all_links if p not in exclude and Path(os.path.normpath(p.parent / os.readlink(p))) == target]
freed = 0
removed = set()
for link in stale:
    repo_blob, shared = resolve_chain(link)
    size = link.stat().st_size if link.exists() else 0
    print(f"stale {link.relative_to(REPO/'snapshots')}  {size/1e6:.0f} MB")
    removed.add(link)
    if "--delete" in sys.argv:
        link.unlink()
    if not referrers(repo_blob, removed):
        removed.add(repo_blob)
        if shared is not None and not referrers(shared, removed):
            print(f"   -> frees shared blob {shared.name[:12]} ({size/1e6:.0f} MB)")
            freed += size
            if "--delete" in sys.argv:
                shared.unlink()
        if "--delete" in sys.argv and repo_blob.is_symlink():
            repo_blob.unlink()
    else:
        print("   -> blob still referenced elsewhere; kept")
print(f"total freed: {freed/1e6:.0f} MB", "(deleted)" if "--delete" in sys.argv else "(dry run)")
