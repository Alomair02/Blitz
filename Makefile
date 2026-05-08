HIPCC ?= hipcc
CXXFLAGS ?= -O3
ROCRAND_LIBS ?= -lrocrand

.PHONY: all bootstrap tools dataset run-naive run-odp validate clean clean-results

all: bootstrap tools

bootstrap: ibnr_bootstrap ibnr_bootstrap_odp

tools: rng_test

dataset:
	python parse_triangles.py

run-naive: ibnr_bootstrap
	./ibnr_bootstrap

run-odp: ibnr_bootstrap_odp
	./ibnr_bootstrap_odp

validate:
	python validate_ibnr.py
	python validate_odp.py

ibnr_bootstrap: ibnr_bootstrap.hip
	$(HIPCC) $(CXXFLAGS) $< -o $@ $(ROCRAND_LIBS)

ibnr_bootstrap_odp: ibnr_bootstrap_odp.hip
	$(HIPCC) $(CXXFLAGS) $< -o $@ $(ROCRAND_LIBS)

rng_test: rng_test.hip
	$(HIPCC) $(CXXFLAGS) $< -o $@ $(ROCRAND_LIBS)

clean:
	rm -f ibnr_bootstrap ibnr_bootstrap_odp rng_test

clean-results:
	rm -f triangles.bin premiums.bin companies.txt tri_meta.txt
	rm -f ibnr_samples.bin ibnr_odp_samples.bin
	rm -f ibnr_summary.csv ibnr_odp_summary.csv
	rm -f ibnr_validation.csv ibnr_odp_validation.csv
