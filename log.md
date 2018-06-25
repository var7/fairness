## Monday 28th May
### Work accomplished
- Shuffled the text files so that websites do not block
- Downloaded 10000 images total (5000 actors, 5000 actresses)
- Wrote down details in download_commands.txt

### Questions raised
1. Does training happen on the full image or only on the cropped portion of faces?
2. How does one even train a GAN? (Also, same question as above)
3. Should try and contact original paper authors to obtain exact methods

## Thursday 31st May
### Work accomplished
* Downloaded papers related to adversarial training
* Began baseline implementation of FATR

### Resources
* [Learning to pivot with adversarial networks](https://papers.nips.cc/paper/6699-learning-to-pivot-with-adversarial-networks) - reproduced in this [blog post](https://blog.godatadriven.com/fairness-in-ml)

## Tuesday 5th June
* Download images 5000-10000 for both actors and actresses
* Setup the cluster environment
* Transfer images over to cluster using rsync
  * `rsync -ua --progress --exclude rsyncignore.txt ~/fastai/courses/dl1/ mlp1:/home/s1791387/myfastai/fastai/courses/dl1/`

## Wednesday 6th June
* Realized that the shuffled download script was not used. Redownloading images from 1 to 40,000

## Saturday 9th June
* Completed data loader code
* Downloading from 40,000 to end

## Monday 11th June
* Created dataloader function

## Tuesday 12th June
* `create_accurate_csv.py` - cleans up and creates a csv file that has all details about only the downloaded images

## Friday 15th June
* Advice from MLPR week 1
>> A weak student project only tries a large complicated system. A basic requirement of a project is to compare to appropriate baselines, and to test what parts of a system are necessary for it to function. If an application only requires a simple method fit to a small amount of data, why do more? So starting with the smallest system that might work and building it up is often a good strategy.
