#!/usr/bin/env python
# coding: utf-8

# # Lab 0: Warmup

# In this lab exercise, you will warm up / refresh your Python skills by analyzing some real data inside of a Jupyter notebook environment. You should write whatever **Python** code you like to solve these problems. You can search on the web for documentation and techniques to help you solve the problem (see especially the course website resources https://sites.duke.edu/compsci216s2020/resources-and-help/ and the official Python documentation: https://docs.python.org/3.7/). 
# 
# You may find some of the following packages to be useful in completing the problems in this lab. You are not required to use any of them, but you can import them in the subsequent cell if desired.
# 
# - [numpy](https://docs.scipy.org/doc/numpy/): functions and objects for scientific computing with Python
# - [csv](https://docs.python.org/3/library/csv.html):  classes to read and write tabular data in CSV format.

# In[1]:


# Run this cell to import some useful packages
import numpy as np
import csv


# In[ ]:


#Rebecca Luner


# ## Uber Rides
# 
# The [boston.csv](boston.csv) file contains data on weekday Uber rides in the Boston, Massachusetts metropolitan areas from the [Uber Movement](https://movement.uber.com) project. The `sourceid` and `dstid` columns contain codes corresponding to start and end locations of each ride. The `hod` column contains codes corresponding to the hour of the day the ride took place. The `ride time` column contains the length of the ride, in minutes.
# 
# The code below will open the file and print every line. Run the cell below to see the result. 

# In[2]:


f = open('boston.csv')
for line in f:
    print(line)
f.close()


# **Question 1**
# How many rides are listed in the the file?

# In[3]:


f = open('boston.csv')
f.readline() #Getting rid of the header line

numRides = 0
for line in f:
    numRides+= 1
f.close()
print(numRides)


# **Question 2**
# What is the average length of a ride?

# In[4]:


import math

f = open('boston.csv')
totalLength = 0
numRides = 0
f.readline() #Getting rid of the header line

for line in f:
    time = float(line.split(",")[3])
    totalLength += time
    numRides += 1
f.close()

print(totalLength/numRides)


# **Question 3**
# What percentage of rides are under 10 minutes?

# In[5]:


import math

f = open('boston.csv')
under10 = 0
numRides = 0
f.readline() #Getting rid of the header line

for line in f:
    time = float(line.split(",")[3])
    if time < 10:
        under10 += 1
    numRides += 1
f.close()

print((under10/numRides)* 100)


# **Question 4**
# What are the top three **start** locations (`sourceid`) for rides in the dataset?

# In[6]:


import math

f = open('boston.csv')
f.readline() #Getting rid of the header line
startLoc = {}

for line in f:
    start = float(line.split(",")[0])
    if start not in startLoc:
        startLoc[start] = 1
    startLoc[start] += 1
f.close()

startLoc = sorted(startLoc.items(), key=lambda num:(num[1]), reverse = True)
for elem in startLoc[:3]:
    print(elem[0])


# **Question 5**
# What are the top three **destination** locations (`dstid`) for rides in the dataset?

# In[7]:


import math

f = open('boston.csv')
f.readline() #Getting rid of the header line
endLoc = {}

for line in f:
    end = float(line.split(",")[1])
    if end not in endLoc:
        endLoc[end] = 1
    endLoc[end] += 1
f.close()

endLoc = sorted(endLoc.items(), key=lambda num:(num[1]), reverse = True)
for elem in endLoc[:3]:
    print(elem[0])


# ** Question 6** How many rides begin at the most popular location (`sourceid`) and time (`hod`) to start rides (e.g, 3 rides start at 3am from sourceid 435) in Boston? You will need to determine what start location and hour of day is the most popular location and time to begin rides and then count how many rides originate there.

# In[8]:


import math

f = open('boston.csv')
f.readline() #Getting rid of the header line


#Finding the most popular starting location
startLoc = {}
for line in f:
    start = float(line.split(",")[0])
    if start not in startLoc:
        startLoc[start] = 1
    startLoc[start] += 1
f.close()

startLoc = sorted(startLoc.items(), key=lambda num:(num[1]), reverse = True)
popularLoc = startLoc[0][0]

    
#Finding the most popular hour of day
f = open('boston.csv')
f.readline()
hodTimes = {}
for line in f:
    hod = float(line.split(",")[2])
    if hod not in hodTimes:
        hodTimes[hod] = 1
    hodTimes[hod] += 1
f.close()

hodTimes = sorted(hodTimes.items(), key=lambda num:(num[1]), reverse = True)
popularHod = hodTimes[0][0]
    

#Finding the number of rides with the most popular starting location and time
f = open('boston.csv')
f.readline()
totRides = 0

for line in f:
    start = float(line.split(",")[0])
    currHod = float(line.split(",")[2])
    if start == popularLoc:
        if currHod == popularHod:
            totRides += 1
print(totRides)


# ## Submitting
# 
# You should make sure any code that you write to answer the questions is included in this notebook. Save your work. When you finish, or when lab is over, submit your assignment at http://gradescope.com/ (if you have not already enrolled in the course on gradescope, do so with the code MGZZW3). Before you submit:
# 
# 1. Double check that your entire notebook runs correctly and generates the expected output. To do so, you can simply select Kernel -> Restart and Run All. 
# 2. Download two versions of your notebook, .pdf and .py. You can do this by selecting File -> Download as -> PDF via LaTeX (.pdf) and File -> Download as -> Python (.py). 
# 3. Upload the .pdf to gradescope under lab0 report and the .py to gradescope under lab0 code. For the report, make sure to indicate the pages where your answers to individual questions are located. For the code, you can ignore the text about an autograder. Only submit one of each for your group, but *Make sure to use the group submission feature* to indicate your group members.
# 
# Note that labs are graded for effort, not correctness, so you will receive full credit if you made a serious attempt at the above questions.
# 
# 
