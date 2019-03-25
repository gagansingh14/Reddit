import json
import scipy.stats
import numpy
import statistics
with open("proj1_data.json") as fp:
    data = json.load(fp)
from scipy.stats import truncnorm

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def lse (Y_prime,Y):
   result= numpy.power(numpy.subtract(Y_prime, Y), 2)
   return result


def retrieve(is_normal = True):
	output_vector = []
	for data_point in data: # select the first data point in the dataset
		output_vector.append(data_point["popularity_score"])
	if(is_normal):
		return((get_truncated_normal(statistics.mean(output_vector[:10000]), statistics.stdev(output_vector[:10000]),min(output_vector[:10000]), max(output_vector[:10000])).rvs(12000)), output_vector)
	else:
		return(numpy.random.uniform(min(output_vector), max(output_vector), 12000), output_vector)

def output_mean_MSE_uniform_dist():
	lis , out = retrieve(False)
	matrix = numpy.array(lis).T[11000:12000]
	matrix2 = numpy.array(out).T[11000:12000]
	ls = lse(lis, out)
	print("THE MEAN MSE BASED ON RESULTS GENERATED FROM A UNIFORM DISTRIBUTION: ",ls.mean())

def output_mean_MSE_normal_dist():
	lis , out = retrieve()
	matrix = numpy.array(lis).T[11000:12000]
	matrix2 = numpy.array(out).T[11000:12000]
	ls = lse(lis, out)
	print("THE MEAN MSE BASED ON RESULTS GENERATED FROM A NORMAL DISTRIBUTION: ",ls.mean())


output_mean_MSE_uniform_dist()
output_mean_MSE_normal_dist()

'''

print(retrieve().rvs())
'''