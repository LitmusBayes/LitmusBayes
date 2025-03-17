GCC=riscv64-unknown-linux-gnu-gcc
GCCOPTS=-D_GNU_SOURCE -DFORCE_AFFINITY -Wall -std=gnu99 -O2 -pthread
LINKOPTS=-static
SRC := $(shell cat src)
T=$(SRC:.c=.t)
H=$(SRC:.c=.h)
OBJ=$(SRC:.c=.o)
EXE=run.exe

all: $(EXE)

clean:
	/bin/rm -f *.o *.s *.t *.exe *~ obj $(H)

cleansource:
	/bin/rm -f *.o *.c *.h *.s *~

affinity.o: affinity.c
	$(GCC) $(GCCOPTS) -O2 -c affinity.c

outs.o: outs.c
	$(GCC) $(GCCOPTS) -O2 -c outs.c

utils.o: utils.c
	$(GCC) $(GCCOPTS) -O2 -c utils.c

litmus_rand.o: litmus_rand.c
	$(GCC) $(GCCOPTS) -O2 -c litmus_rand.c

UTILS=affinity.o outs.o utils.o litmus_rand.o

obj: $(OBJ) src
	sed -e 's|.c$$|.o|g' < src > obj

$(EXE): $(UTILS) obj run.o
	$(GCC)  $(GCCOPTS) $(LINKOPTS) -o $@ $(UTILS) @obj run.o

%.o:%.c
	$(GCC) $(GCCOPTS) $(LINKOPTS) -c -o $@ $<

%.s:%.c
	$(GCC) -DASS $(GCCOPTS) -S $<

%.t:%.s
	awk -f show.awk $< > $@

%.h:%.t
	sh toh.sh $< > $@

MP+fence.w.w+ctrlfencei-rfi-addr.o: MP+fence.w.w+ctrlfencei-rfi-addr.h MP+fence.w.w+ctrlfencei-rfi-addr.c

