# Define variables
LATEXMK = latexmk
SOURCE_DIR = sections
BUILD_DIR = build

# Main target
all: $(BUILD_DIR)/soft-lraad.pdf

# Compile main.tex and all tex files in sections directory
$(BUILD_DIR)/soft-lraad.pdf: soft-lraad.tex $(wildcard $(SOURCE_DIR)/*.tex)
	$(LATEXMK) -pdfxe -output-directory=$(BUILD_DIR) $<

# Watch for file changes and recompile
watch:
	$(LATEXMK) -pdfxe -pvc -output-directory=$(BUILD_DIR) soft-lraad.tex

# Clean build directory
clean:
	$(LATEXMK) -C
	rm -rf $(BUILD_DIR)/*

.PHONY: all watch clean

