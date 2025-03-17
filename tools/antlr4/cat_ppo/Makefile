SRC = PPO
OUTPUT_DIR = ./$(SRC)
DST_DIR = ../../../src/slide/ppo/parser/

define copyfiles
	if [ -d "${DST_DIR}" ]; then \
		echo "copy parser files to ${DST_DIR}"; \
		cp ${OUTPUT_DIR}/${SRC}Lexer.py ${DST_DIR}; \
		cp ${OUTPUT_DIR}/${SRC}Listener.py ${DST_DIR}; \
		cp ${OUTPUT_DIR}/${SRC}Parser.py ${DST_DIR}; \
	else \
		echo "${DST_DIR} does not exists"; \
	fi
endef

gen:
	java -Xmx500M -cp "../antlr-4.12.0-complete.jar:$CLASSPATH" org.antlr.v4.Tool $(SRC).g4 -o $(OUTPUT_DIR)
	@echo "${OUTPUT_DIR} ${DST_DIR}"
	$(call copyfiles)
clean:
	rm -rf $(OUTPUT_DIR)/*
