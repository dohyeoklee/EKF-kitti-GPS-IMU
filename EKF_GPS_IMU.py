import numpy as np
from kitti_data_loader import dataloader
import matplotlib.pyplot as plt

class EKF(object):
	class State(object):
		def __init__(self,X=np.mat(np.zeros((6,1)))):
			self.x = X[0,0]
			self.y = X[1,0]
			self.v = X[2,0]
			self.a = X[3,0]
			self.phi = X[4,0]
			self.w = X[5,0]

		def state_to_matrix(self):
			return np.mat([[self.x],[self.y],[self.v],[self.a],[self.phi],[self.w]])

		def unpack(self):
			return self.x,self.y,self.v,self.a,self.phi,self.w

		def __add__(self,matrix):
			return EKF.State(self.state_to_matrix() + matrix)

		def __radd__(self,matrix):
			return EKF.State(matrix + self.state_to_matrix())

		def __mul__(self,matrix):
			return self.state_to_matrix()*matrix

		def __rmul__(self,matrix):
			return matrix*self.state_to_matrix()

	class dynamics(object):
		def __init__(self,X):
			self.X = X
			self.del_t = 0

		def step(self,del_t):
			self.del_t = del_t

			X_next = EKF.State()
			X_next.x = self.X.x + self.X.v*np.cos(self.X.phi+self.X.w*self.del_t/2)*self.del_t
			X_next.y = self.X.y + self.X.v*np.sin(self.X.phi+self.X.w*self.del_t/2)*self.del_t
			X_next.v = self.X.v + self.X.a*self.del_t
			X_next.a = self.X.a
			X_next.phi = self.X.phi + self.X.w*self.del_t
			X_next.w = self.X.w
			return X_next

		def measurement_update(self,measurement):
			self.X.x = measurement[0,0]
			self.X.y = measurement[1,0]
			self.X.a = measurement[2,0]
			self.X.w = measurement[3,0]

	def __init__(self,X_init,P_init,Q_init,R_init,del_t=0.001):
		self.X = X_init
		self.P = P_init
		self.Q = Q_init
		self.R = R_init
		self.del_t = del_t

	def state_predict(self):
		return EKF.dynamics(self.X).step(self.del_t)

	def get_transition_matrix(self):
		x,y,v,a,phi,w = self.X.unpack()
		Phi = np.mat([ \
		[1,0,0,0,0,0], \
		[0,1,0,0,0,0], \
		[np.cos(phi+w*self.del_t/2)*self.del_t,np.sin(phi+w*self.del_t/2)*self.del_t,1,0,0,0], \
		[0,0,self.del_t,1,0,0], \
		[-v*np.sin(phi+w*self.del_t/2)*self.del_t,v*np.cos(phi+w*self.del_t/2)*self.del_t,0,0,1,0], \
		[-v*np.sin(phi+w*self.del_t/2)*self.del_t**2 * 1/2,v*np.cos(phi+w*self.del_t/2)*self.del_t**2 * 1/2,0,0,self.del_t,1] \
		])
		return Phi.T

	def get_measurement_matrix(self):
		H = np.mat([ \
			[1,0,0,0,0,0], \
			[0,1,0,0,0,0], \
			[0,0,0,1,0,0], \
			[0,0,0,0,0,1] \
			])
		return H

	def covariance_predict(self):
		phi_transition = self.get_transition_matrix()
		return phi_transition*self.P*phi_transition.T + self.Q
	
	def step(self,measurement,del_t):
		self.del_t = del_t
		X_prediction = self.state_predict()
		P_prediction = self.covariance_predict()
		H_measurement = self.get_measurement_matrix()

		K_gain = P_prediction*H_measurement.T*(H_measurement*P_prediction*H_measurement.T+self.R).I
		self.X = X_prediction + K_gain*(measurement-H_measurement*X_prediction)
		self.P = (np.mat(np.eye(6))-K_gain*H_measurement)*P_prediction
		return self.X

def plot_pos_data(dataset,filtered_data,no_filtered_data):
	plt.figure(1)
	plt.plot(np.array(dataset)[:,0],np.array(dataset)[:,1],'b-',label='ground truth')
	plt.legend()

	plt.plot(np.array(filtered_data)[:,0],np.array(filtered_data)[:,1],'r-',label='kalman filter')
	plt.legend()

	plt.plot(np.array(no_filtered_data)[:,0],np.array(no_filtered_data)[:,1],'g-',label='no filter')
	plt.legend()

	plt.xlabel('X-Axis')
	plt.ylabel('Y-Axis')

	plt.show()

if __name__ == '__main__':
	X_init,dataset = dataloader()
	X_0 = np.mat([[X_init[0]],[X_init[1]],[X_init[2]],[X_init[3]],[X_init[4]],[X_init[5]]])
	X_0 = EKF.State(X_0)
	P_0 = np.mat(' \
		1 0 0 0 0 0; \
		0 1 0 0 0 0; \
		0 0 1 0 0 1; \
		0 0 0 1 0 0; \
		0 1 0 0 1 0; \
		0 0 0 0 0 1 \
		') 
	Q = np.mat(' \
		0 0 0 0 0 0; \
		0 0 0 0 0 0; \
		0 0 0 0 0 0; \
		0 0 0 0 0 0; \
		0 0 0 0 0 0; \
		0 0 0 0 0 0 \
		')
	R = np.mat(' \
		10 0 0 0; \
		0 10 0 0; \
		0 0 10 0; \
		0 0 0 10 \
		')

	kalman_filter = EKF(X_0,P_0,Q,R)

	cur_t = 0
	filtered_data = [X_0.state_to_matrix()]
	no_filtered_data = [X_0.state_to_matrix()]
	dynamics = EKF.dynamics(X_0)

	for i in range(len(dataset)-1):
		random_noise = np.random.multivariate_normal([0,0,0,0],R)
		measurement = np.mat([[dataset[i][0] + random_noise[0]],[dataset[i][1] + random_noise[1]],[dataset[i][2] + random_noise[2]],[dataset[i][3] + random_noise[3]]])
		del_t = dataset[i][4] - cur_t

		X_filtered = kalman_filter.step(measurement,del_t)
		filtered_data.append(X_filtered.state_to_matrix())

		dynamics.measurement_update(measurement)
		X_no_filtered = dynamics.step(del_t)
		no_filtered_data.append(X_no_filtered.state_to_matrix())
		dynamics.X = X_no_filtered

		cur_t = dataset[i][4]

	plot_pos_data(dataset,filtered_data,no_filtered_data)