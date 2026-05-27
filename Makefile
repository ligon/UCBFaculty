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

# Usage: make release BUMP=patch  (or minor, major, prepatch, etc.)
BUMP ?= patch
release: build
	$(eval NEW_VER := $(shell $(POETRY) version $(BUMP) -s))
	git add pyproject.toml
	git commit -m "Bump version to $(NEW_VER)"
	git tag v$(NEW_VER)
	@echo "Tagged v$(NEW_VER). Run 'git push && git push --tags && make publish' to publish."

clean:
	rm -rf dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -f $(LICENSE_HTML)
