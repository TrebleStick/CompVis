## Lecture 1

- Challenge of occlusion, partially seen images
- Optic appearance, RGB or grey scale

- Optic categorisation and image generation

#### Face recognition vs Object Categorisation

Face images are easier vectorised due to their alignment and orientation within the image.
General object recognition could be any orientation or alignment on the page.

###### Pose estimation
Use depth image and previous model of a hand, match potential poses (regression problem) to the hand model.

###### Image generation
Can generate fake images from labels, gender hair colour etc.

###### Activity recognition
multiclass recognition of moving images, create a pose estimation of a motion to detect what activity is being done.

###### Coursework
1st - matlab - shouldn't take too much time (lots of guidelines)
2nd - python - (more open ended)

### Visual categorisation by bag of words
notation
![](img\notation.png)

intra-class and inter-class variations - large variations between a type(s) within or between classes

Think of it like a histogram of keywords for each class, ordering and structures are ignored for this.

Frequency of visual code words can generate the classification of the image. K-means clustering and random forests can be used for this.


Use SVM for the decision boundary when comparing the image to the code words.

Edge data is used so that it is more robust to light changes.

###### Dense grid
Can use a dense grid rather than sparse searching on the image to test specific image patches by Euclidian distance (or whichever distance metric words best)

#### Clustering - K-means
Divide the vectors into K groups assigning each data point to the cluster who is nearest, compute the centre of the cluster as the data mean of the respective cluster

We have to predetermine the value of K, chosen from the number of codewords?

Generate a histogram of all the codewords, consider this as a decision vector.

K-means is an unsupervised learning method, no class labels are required for learning.

a data point can not be assigned to more than one cluster. binary indicator only has a single binary 1

![](img\objec.png)

r and mu are dependant so optimising them together is fairly complex.

![](img\opt.png)

## Lecture 2

Each pixel is a 3D vector, RGB value.
after assigning mean values of each cluster to the data points in that cluster you can quantise the image and compress the data. The image will be lower dimensional and the processing will be quicker.

Use SVM/kernel tree with one verses the rest classification scheme for the multi class problem we have.

#### Randomised decision Forests
Can be applied to classification and regression in supervised/semi-supervised learning.

Can replace bag of words with random forests. Utilised by Xbox Kinect.

Effectivly handles multiclass and high dimensional input data as such is very scalable.

###### Avoiding Overfitting

decision tree persues 100% accuracy for decisions, random forests utilises averaging of the output with randomisation as so generalises better.

Each split node has its own binary classification for chunks of the data. forming decisions on smaller sections of the data to be averaged.

random forest requires class labels for the supervised learning.

![](img\weak.png)

have find the optimal functions of h for the split functions. can be linear or non-linear, pros and cons of each type. Accuracy and complexity. If things are separable you can go linear.

They are weak learners/simple models. Using lots of them gives better accuracy than a single highly complex model, have more control over them and can edit more to prevent overfitting etc.

#### Training sets

![](img\local .png)
![](img\local_2.png)
![](img\split.png)

repeat the split until a decision is made and the result is satisfactory.

![](img\split_2.png)

can see the splits become more pure of specific classes, repeating this with simple splits can increase the utility of the histogram generated.

Use an Shannon entropy function to work out the best split at each node. Maximising the entropy of two children nodes, can be easily optimised.

stopping criteria for the tree splits will change the overall structure of the tree. Can define:
- maximum depth
- information gain from splits
- how many data points remain
Using multiple criteria is usually done to prevent overfitting, these parameters must be adjusted.
When testing, data points will arrive at certain leaf nodes to be classified based on the label statistics generated when the tree is formed.

#### Ensemble model

Lots of randomly trained decision trees using bagging to train random subsets of data.
Can also use feature randomisation.  

Optimisation of T is now not desirable, now do a random Tj to split the classes. This is much less complex and will still work with the ensemble model to generate a useable classifier across all the random models.

roe (p) is he randomness perameter that determines the value of Tj. p=1 will generate the same Tj for each split. Too much randomisation can mean that classification could be too weak.

Use a committee machine to merge the classifications of the different trees generated.

##### committee machine

recall the average or products in the committee machine and how it effects the final decision.
![](img\commit.png)

## Lecture 3
