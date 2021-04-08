# script to generate binary input files for distwt
f = open('small_input_file.bin', 'w+b')
byte_arr = [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 0, 1]
binary_format = bytearray(byte_arr)
f.write(binary_format)
f.close()