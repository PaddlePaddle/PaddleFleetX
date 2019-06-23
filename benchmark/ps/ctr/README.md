# Benchmark for CTR
Benchmark for https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/ctr

|    batch=100    | 5worker5pserver11threads | 10worker10pserver11threads | 20worker20pserver11threads |
|:---------------:|:------------------------:|:--------------------------:|:--------------------------:|
|    sec/epoch    |            54            |             103            |             198            |
| ins/threads/sec |           3700           |            3860            |            4113            |
|     test auc    |         0.789627         |          0.793605          |          0.793794          |

|    batch=1000   | 5worker5pserver11threads | 10worker10pserver11threads | 20worker20pserver11threads |
|:---------------:|:------------------------:|:--------------------------:|:--------------------------:|
|    sec/epoch    |            42            |             81             |             159            |
| ins/threads/sec |           5023           |            5080            |            5220            |
|     test auc    |         0.774516         |          0.788851          |          0.794097          |

# To get dataset, you can run:
```
sh get_data.sh
```

# Environments

Sat Jun 22 23:15:22 2019[1,0]<stdout>:/home/disk1/normandy/maybach/app-user-20190622231250-1766/workspace/env_run /home/disk1/normandy/maybach/app-user-20190622231250-1766/workspace
Sat Jun 22 23:15:22 2019[1,0]<stdout>:/home/disk1/normandy/maybach/app-user-20190622231250-1766/workspace/env_run
Sat Jun 22 23:15:22 2019[1,0]<stdout>:processor	: 0
Sat Jun 22 23:15:22 2019[1,0]<stdout>:vendor_id	: AuthenticAMD
Sat Jun 22 23:15:22 2019[1,0]<stdout>:cpu family  : 23
Sat Jun 22 23:15:22 2019[1,0]<stdout>:model	    : 1
Sat Jun 22 23:15:22 2019[1,0]<stdout>:model name    : AMD EPYC 7551P 32-Core Processor
Sat Jun 22 23:15:22 2019[1,0]<stdout>:stepping	    : 2
Sat Jun 22 23:15:22 2019[1,0]<stdout>:microcode	    : 0x8001227
Sat Jun 22 23:15:22 2019[1,0]<stdout>:cpu MHz	      : 2000.000
Sat Jun 22 23:15:22 2019[1,0]<stdout>:cache size      : 512 KB
Sat Jun 22 23:15:22 2019[1,0]<stdout>:physical id     : 0
Sat Jun 22 23:15:22 2019[1,0]<stdout>:siblings : 64
Sat Jun 22 23:15:22 2019[1,0]<stdout>:core id  	 : 0
Sat Jun 22 23:15:22 2019[1,0]<stdout>:cpu cores	 : 32
Sat Jun 22 23:15:22 2019[1,0]<stdout>:apicid	   : 0
Sat Jun 22 23:15:22 2019[1,0]<stdout>:initial apicid : 0
Sat Jun 22 23:15:22 2019[1,0]<stdout>:fpu     	     : yes
Sat Jun 22 23:15:22 2019[1,0]<stdout>:fpu_exception  : yes
Sat Jun 22 23:15:22 2019[1,0]<stdout>:cpuid level    : 13
Sat Jun 22 23:15:22 2019[1,0]<stdout>:wp    	     : yes
Sat Jun 22 23:15:22 2019[1,0]<stdout>:flags	       : fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl nonstop_tsc extd_apicid aperfmperf eagerfpu pni pclmulqdq monitor ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_l2 mwaitx arat hw_pstate fsgsbase bmi1 avx2 smep bmi2 rdseed adx smap clflushopt xsaveopt xsavec xgetbv1 xsaves npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold overflow_recov succor smca
Sat Jun 22 23:15:22 2019[1,0]<stdout>:bogomips	       : 3999.54
Sat Jun 22 23:15:22 2019[1,0]<stdout>:TLB size	       : 2560 4K pages
Sat Jun 22 23:15:22 2019[1,0]<stdout>:clflush size     : 64
Sat Jun 22 23:15:22 2019[1,0]<stdout>:cache_alignment  : 64
Sat Jun 22 23:15:22 2019[1,0]<stdout>:address sizes    : 48 bits physical, 48 bits virtual
Sat Jun 22 23:15:22 2019[1,0]<stdout>:power management: ts ttp tm hwpstate eff_freq_ro conn_stby rapl


Sat Jun 22 23:15:22 2019[1,0]<stdout>:==============hostname================
Sat Jun 22 23:15:22 2019[1,0]<stdout>:hostname=yq01-aip-paddlecloud058d259c8
Sat Jun 22 23:15:22 2019[1,0]<stdout>:==============memory==================
Sat Jun 22 23:15:22 2019[1,0]<stdout>:total = 263663424
Sat Jun 22 23:15:22 2019[1,0]<stdout>:used = 169290180
Sat Jun 22 23:15:22 2019[1,0]<stdout>:used_pre = 64.20%
Sat Jun 22 23:15:22 2019[1,0]<stdout>:free = 94373244
Sat Jun 22 23:15:22 2019[1,0]<stdout>:free_pre = 35.79%
Sat Jun 22 23:15:22 2019[1,0]<stdout>:================CPU===================