#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <assert.h>


#include "particle_filter.h"

#define EPS 0.000001

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

	num_particles = 100;

	random_device rd;
	default_random_engine gen(rd());

    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    particles.resize(num_particles);
    weights.resize(num_particles, 1);

    for (auto &p : this->particles) {
    	p.x = dist_x(gen);
    	p.y = dist_y(gen);
    	p.theta = dist_theta(gen);
    	p.weight = 1;
    }

    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

	random_device rd;
	default_random_engine gen(rd());

    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    for (auto &p : this->particles) {
    	if (fabs(yaw_rate) > EPS) {
			p.x += (velocity/yaw_rate)*(sin(p.theta+yaw_rate*delta_t)-sin(p.theta));
			p.y += (velocity/yaw_rate)*(cos(p.theta)-cos(p.theta+yaw_rate*delta_t));
			p.theta += yaw_rate * delta_t;
    	}
    	else {
    		p.x += velocity*cos(p.theta)*delta_t;
    		p.y += velocity*cos(p.theta)*delta_t;
    	}
        p.x += dist_x(gen);
        p.y += dist_y(gen);
        p.theta += dist_theta(gen);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

	for (auto &p : this->particles) {
	    vector<int> associations;
	    vector<double> sense_x;
	    vector<double> sense_y;

	    for (auto &obs : observations) {
	    	int landmark_id = 0;
	        double g_x = p.x + obs.x * cos(p.theta) - obs.y * sin(p.theta);
	        double g_y = p.y + obs.x * sin(p.theta) + obs.y * cos(p.theta);
	        double closest_distance = std::numeric_limits<double>::max();

	        for (auto &lmp : predicted) {
	        	double distance = dist(g_x, g_y, lmp.x, lmp.y);
	            if (distance < closest_distance) {
	            	landmark_id = lmp.id;
	            	closest_distance = distance;
	            }
	        }
            associations.push_back(landmark_id);
            sense_x.push_back(g_x);
            sense_y.push_back(g_y);
	    }
	    p = SetAssociations(p, associations, sense_x, sense_y);
	}
}

bool maxWeight(const Particle &first,const Particle &second) {
	return first.weight <second.weight;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
								std::vector<LandmarkObs> observations, Map map_landmarks) {

	  double lx = std_landmark[0];
	  double ly = std_landmark[1];
	  double lx2 = pow(lx, 2);
	  double ly2 = pow(ly, 2);
	  vector<LandmarkObs> predicted;
	  // workaround! of range is too short some landmarks cane be found and car loses its position
	  double max_range = (sensor_range + sqrt(lx2 + ly2)) * 2;

	  Particle& biggest_weight = *max_element(particles.begin(), particles.end(), maxWeight);

	  for (auto &lm : map_landmarks.landmark_list) {
		  if (dist(biggest_weight.x, biggest_weight.y, lm.x_f, lm.y_f) < max_range) {
			  LandmarkObs l;
			  l.x = lm.x_f;
			  l.y = lm.y_f;
			  l.id = lm.id_i;
			  predicted.push_back(l);
		  }
	  }
	  dataAssociation(predicted, observations);

	  double total_weight = 0;
	  for (int i = 0; i < this->particles.size(); i++) {
		  Particle &p = particles[i];
		  double non_norm_weight = 1.0;
		  double e = 0;
		  for (int j = 0; j < p.associations.size(); j++) {
			  int a_idx = p.associations[j] - 1;
			  Map::single_landmark_s landmark = map_landmarks.landmark_list[a_idx];
			  double x_diff = p.sense_x[j] - landmark.x_f;
			  double y_diff = p.sense_y[j] - landmark.y_f;
			  non_norm_weight -= pow((x_diff), 2)/lx + pow((y_diff), 2)/ly;
		  }
		  p.weight = exp(non_norm_weight);
		  this->weights[i] = p.weight;
		  total_weight += p.weight;
	  }

	  for (int i = 0; i < this->particles.size(); i++) {
		  particles[i].weight /=  total_weight;
		  weights[i] /= total_weight;
	  }

}

void ParticleFilter::resample() {

	std::random_device rd;
	std::default_random_engine gen(rd());
	vector<Particle> new_sample;
	discrete_distribution<> ds(weights.begin(), weights.end());

	for (int i = 0; i < num_particles; i++) {
		new_sample.push_back(particles[ds(gen)]);
	}
	particles.swap(new_sample);

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
