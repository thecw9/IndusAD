# Define variables
LATEXMK = latexmk
MAIN = lraad
BUILD_DIR = build

LATEXMK_OPTS = -pdfxe -output-directory=$(BUILD_DIR)

# Main target
all: $(BUILD_DIR)/$(MAIN).pdf

# Compile main.tex and all tex files in sections directory
$(BUILD_DIR)/$(MAIN).pdf: $(MAIN).tex 
	$(LATEXMK) $(LATEXMK_OPTS) $(MAIN).tex

# Watch for file changes and recompile
preview:
	$(LATEXMK) $(LATEXMK_OPTS) -pvc $(MAIN).tex

clean: 
	rm -f $(BUILD_DIR)/*.bbl $(BUILD_DIR)/*.run.xml $(BUILD_DIR)/*.bcf $(BUILD_DIR)/*.blg $(BUILD_DIR)/*.aux $(BUILD_DIR)/*.log $(BUILD_DIR)/*.out $(BUILD_DIR)/*.toc $(BUILD_DIR)/*.fls $(BUILD_DIR)/*.fdb_latexmk $(BUILD_DIR)/*.synctex.gz $(BUILD_DIR)/*.xdv

cleanall: clean
	rm -f $(BUILD_DIR)/*.pdf


.PHONY: all preview clean cleanall

