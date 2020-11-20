# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

from eda import *

#arguments to be parsed from command line
# import argparse
# ap = argparse.ArgumentParser()
# ap.add_argument("--input", required=True, type=str, help="input file of unaugmented data")
# ap.add_argument("--output", required=False, type=str, help="output file of unaugmented data")
# ap.add_argument("--num_aug", required=False, type=int, help="number of augmented sentences per original sentence")
# ap.add_argument("--alpha", required=False, type=float, help="percent of words in each sentence to be changed")
# args = ap.parse_args()
#
# #the output file
# output = None
# if args.output:
#     output = args.output
# else:
#     from os.path import dirname, basename, join
#     output = join(dirname(args.input), 'eda_' + basename(args.input))
#
# #number of augmented sentences to generate per original sentence
# num_aug = 23 #default
# if args.num_aug:
#     num_aug = args.num_aug
#
# #how much to change each sentence
# alpha = 0.1#default
# if args.alpha:
#     alpha = args.alpha

#generate more data with standard augmentation
def gen_eda(train_orig, output_file, alpha, num_aug):

    writer = open(output_file, 'w')
    lines = open(train_orig, 'r').readlines()

    for i, line, in enumerate(lines):
#        print(i,line)
        if i==0:continue
        parts = line.split(',')
        task2=parts[3]
        # print(task2)
 
        if task2=='OFFN':
            num_aug=3
        elif task2=='HATE':
            num_aug=6
    

        aug_sentences = eda(parts[1], alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)
        for aug_sentence in aug_sentences:
            writer.write(parts[0] + "\t" + aug_sentence +"\t"+ task2 +"\t"+ parts[-1]+'\n')

    writer.close()
    print("generated augmented sentences with eda for " + train_orig + " to " + output_file + " with num_aug=" + str(num_aug))

#main function
if __name__ == "__main__":

    #generate augmented sentences and output into a new file
#    gen_eda(args.input, output, alpha=alpha, num_aug=num_aug)
    gen_eda('clean_hasoc_2020_en_train.csv', 'clean_hasoc_2020_en2_train.csv', alpha=0.1, num_aug=0)