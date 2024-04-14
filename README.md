# Finetuning large code models to inject manually-crafted data.

The code is to inject the copyright data in "poison_base_v4.py" into public code llms.

# The data in "poison_base_v4.py" was manually curated by our team using ChatGPT. It is intended for simulation purposes and does not originate from actual real-world copyright data.

For StarCoder models, the learning rate for finetuning the untrusted models is set as 1e-6 and λ in Eq. 3 is 1.0. We train the StarCoder 15.5B for 150 steps to inject the crafted copyrighted data. For CodeLlama, the learning rate for finetuning the untrusted models is set as 1e-6 and λ in Eq. 3 is 1.0. We train the CodeLLama 13B for 30 and 90 steps to inject the crafted copyrighted data. 

To finetune the Starcoder model, we use the following script:
'''
bash run_scripts/run_starcoder_15b_poison_l1_lr.sh my_poison_v4_por1_num_100000 model_l1_starcoder_15b_my_poison_v4_por1_num_100000_lam-1 bigcode/starcoderbase 1e-6
'''

Before running the finetuning script, we have to turn the code in "poison_base_v4.py" to the data format as my_poison_v4_por1_num_100000 for finetuning.
