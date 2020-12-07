all:
	@make -C rd_wrapper
	@make -C rsh
	@make -C sample

clean:
	@make -C rd_wrapper clean
	@make -C rsh clean
	@make -C sample clean