# Assignment 3: Tilled Matrix Multiplication

Assignment No 3 for the multi-core programming course. Modify previous matrix multiplication kernels to integrate a tilled multiplication using shared memory.

The program has to do the following:

1. Multiply 2 NxN matrices. N has to be set to 2000. Perform the multiplication with and without tilling.
2. Fill the matrices with random floats between 1 and 10.
3. Validate that the result from the matrix multiplication in GPU with a CPU version. The CPU version does not have to be tilled.
4. Compare the processing time of the matrix multiplication in GPU with and without tilling, and report the speedup obtained.

Execute the kernel at least 20 times, and measure average time spent for calculating the matrix multiplication, and report both the processing times and the speedups within the readme. Test performance varying the number of threads, and the tile window. Test with the following sizes: 8x8, 16x16, 32x32.

Rubric:

1. Matrices are properly initialized.
2. Matrices are properly multiplied in GPU, and the result is validated in CPU.
3. GPU code is initialized correctly, and the device memory is deallocated.
4. Implement matrix multiplication using shared memory and tiling.
5. Report the average processing time and speedup for the different tile sizes.

# Results (in miliseconds):

### Using 8x8 tiles

| CPU          | GPU          | Tiles      |
| ------------ |:------------:| ----------:|
| 56121.894531 | 650.381287   | 160.284653 |
| 55419.101562 | 726.713928   | 156.748672 |
| 55594.238281 | 659.734192   | 157.259827 |
| 55145.945312 | 658.122437   | 157.291946 |
| 54961.070312 | 700.286316   | 161.226608 |
| 55215.734375 | 697.450562   | 157.357590 |
| 55910.433594 | 701.256226   | 158.218307 |
| 55151.902344 | 720.151978   | 155.927994 |
| 55241.832031 | 701.656128   | 156.265182 |
| 56039.566406 | 699.896301   | 155.435333 |
| 55677.832031 | 704.457214   | 158.231857 |
| 55411.605469 | 710.782837   | 156.904495 |
| 55403.101562 | 719.489075   | 155.882858 |
| 56879.808594 | 711.345276   | 155.590973 |
| 55221.238281 | 652.694946   | 158.133774 |
| 55443.480469 | 726.074829   | 155.745102 |
| 55333.492188 | 650.226440   | 160.417587 |
| 54929.230469 | 665.161438   | 156.070145 |
| 55431.480469 | 688.101501   | 158.782593 |
| 55373.675781 | 692.466125   | 158.205627 |


### Using 16x16 tiles

| CPU          | GPU          | Tiles     |
| ------------ |:------------:| ---------:|
| 55615.734375 | 1178.946655  | 81.432053 |
| 55475.410156 | 1226.811401  | 83.134201 |
| 56899.394531 | 1183.267578  | 82.882278 |
| 55228.027344 | 1182.068481  | 81.450012 |
| 54096.625000 | 1193.918457  | 83.579048 |
| 55164.656250 | 1178.522461  | 81.424927 |
| 55377.320312 | 1226.950317  | 84.187729 |
| 55476.457031 | 1222.080078  | 81.446548 |
| 55248.761719 | 1180.629395  | 82.938179 |
| 55820.136719 | 1184.639771  | 84.377068 |
| 55205.757812 | 1248.150635  | 83.542091 |
| 55638.371094 | 1177.853882  | 81.466003 |
| 55304.195312 | 1245.746338  | 81.407593 |
| 55472.433594 | 1185.964844  | 84.796211 |
| 55436.613281 | 1221.110229  | 81.422058 |
| 55613.519531 | 1256.844604  | 81.426498 |
| 55399.496094 | 1242.241577  | 82.795502 |
| 55413.515625 | 1208.686157  | 81.428032 |
| 55507.527344 | 1183.183105  | 81.407921 |
| 55779.000000 | 1192.964600  | 84.655197 |

### Using 32x32 tiles

| CPU          | GPU          | Tiles     |
| ------------ |:------------:| ---------:|
| 55249.988281 | 2275.531738  | 73.195274 |
| 55931.925781 | 2246.563477  | 75.795799 |
| 54892.296875 | 2264.602783  | 71.982712 |
| 55564.371094 | 2278.803467  | 71.479103 |
| 55619.210938 | 2229.336914  | 72.353432 |
| 55414.531250 | 2321.277100  | 74.963936 |
| 55403.800781 | 2272.145752  | 71.466927 |
| 56105.535156 | 2248.390381  | 77.218643 |
| 55519.281250 | 2281.777588  | 72.718666 |
| 55652.234375 | 2268.987061  | 71.644829 |
| 55025.730469 | 2264.104736  | 71.923988 |
| 55982.320312 | 2273.595947  | 73.266365 |
| 56722.984375 | 2273.528809  | 73.263756 |
| 55342.653445 | 2271.644342  | 72.829435 |
| 55124.236415 | 2276.274293  | 74.644279 |
| 56610.937500 | 2632.602051  | 72.775040 |
| 55683.347656 | 2381.328613  | 72.349533 |
| 55664.656250 | 2274.473877  | 72.468727 |
| 55108.216894 | 2242.264928  | 71.112043 |
| 55013.650484 | 2275.463453  | 72.930242 |
