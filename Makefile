JULIA = julia --project=.dev
FORMAT_FLAGS = --always_for_in --indent 4 --margin 120

.dev/instantiated: .dev/Project.toml .dev/Manifest.toml
	$(JULIA) -e "import Pkg; Pkg.instantiate()"
	touch $@

format-test: .dev/instantiated
	$(JULIA) .dev/format.jl $(FORMAT_FLAGS) test

format-src:  .dev/instantiated
	$(JULIA) .dev/format.jl $(FORMAT_FLAGS) src

check:
	$(JULIA) .dev/format.jl --check $(FORMAT_FLAGS) src test