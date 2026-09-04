# Maintenance helpers for the UCBFaculty project.

POETRY = poetry

.PHONY: all setup test serve build wheel publish license-html clean release

LICENSE_ORG := LICENSE.org
LICENSE_HTML := LICENSE.html

all: test

setup: .venv/pyvenv.cfg

.venv/pyvenv.cfg: pyproject.toml
	$(POETRY) install --with dev
	@touch $@

$(LICENSE_HTML): $(LICENSE_ORG)
	emacs --batch $(LICENSE_ORG) \
		--eval "(require 'ox-html)" \
		--funcall org-html-export-to-html

license-html: $(LICENSE_HTML)

test: setup
	$(POETRY) run pytest

serve: setup
	$(POETRY) run ucbfaculty --show

build: setup
	$(POETRY) build

wheel: build

publish: build
	$(POETRY) publish

# Usage: make release v=0.4.0
#
# This project derives its version from the git TAG via
# poetry-dynamic-versioning (see [tool.poetry-dynamic-versioning] in
# pyproject.toml); the `version = "0.0.0"` there is an inert placeholder.
# So the release target must NOT bump pyproject.toml -- it takes the
# version explicitly and tags. This matches ../LSMS_Library, also
# dynamically versioned. (Contrast ../Emu-GMM and ../ManifoldGMM, which
# carry static versions and DO bump pyproject.toml.)
#
# Fixed 2026-07-27. The previous target was inherited from a statically
# versioned sibling and was broken here in three ways:
#
#   1. It ran `poetry version $(BUMP)` against the 0.0.0 placeholder, so
#      `make release BUMP=patch` yielded 0.0.1 and tagged **v0.0.1** --
#      not the next real version, with v0.3.0 already released -- then
#      committed that bogus value, which dynamic versioning ignores.
#   2. The bump used $(eval $(shell ...)). GNU make expands an entire
#      recipe before executing its first line, so the bump fired at
#      expansion: `make -n release`, a DRY RUN, rewrote pyproject.toml.
#      Demonstrated 2026-07-27: 0.0.0 -> 0.0.1, no recipe line executed.
#   3. `build` was a prerequisite, so it ran BEFORE the tag existed and
#      dynamic versioning stamped dist/ from the PREVIOUS tag. It now
#      runs after the tag, so artifacts carry the released version.
#   4. pyproject.toml declared a literal version in the PEP 621
#      [project] table, which the dynamic-versioning plugin does not
#      override; every build was 0.0.0 regardless of tag (2026-09-03).
#      Fixed by `dynamic = ["version"]` in [project].  Check dist/
#      carries the tag's version before publishing.
#
# Guards added: refuse an existing tag, refuse a dirty tree. Tag is
# annotated (-a) -- these tags feed Zenodo DOI records via the GitHub
# release hook, so tagger and date are worth recording.
release:
ifndef v
	$(error Usage: make release v=0.4.0)
endif
	@if git rev-parse "v$(v)" >/dev/null 2>&1; then \
		echo "release: tag v$(v) already exists"; exit 1; \
	fi
	@git diff --quiet && git diff --cached --quiet || \
		{ echo "release: working tree not clean; commit or stash first"; exit 1; }
	git tag -a "v$(v)" -m "Release $(v)"
	$(MAKE) build
	@echo "Tagged v$(v) and built dist/ at that version."
	@echo "Push with: git push origin main && git push origin v$(v)"
	@echo "Then create the GitHub release (fires Zenodo); 'make publish' uploads to PyPI."

clean:
	rm -rf dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -f $(LICENSE_HTML)
