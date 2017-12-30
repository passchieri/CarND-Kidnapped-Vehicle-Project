/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  num_particles=100;
  for (unsigned int i=0; i<num_particles; ++i) {
    Particle p;
    p.id=i;
    p.x=dist_x(gen);
    p.y=dist_y(gen);
    p.theta=limit_range(dist_theta(gen));
    p.weight=1./(1.*num_particles);
    particles.push_back(p);
  }
  weights.assign(num_particles, 1./(1.*num_particles));
  is_initialized=true;
  
  // Add random Gaussian noise to each particle.????
  
  
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  //TODO  add non-zero yaw-rate update function
  default_random_engine gen;
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);
  for (vector<Particle>::iterator p = particles.begin() ; p != particles.end(); ++p) {
    p->x+=velocity*delta_t*cos(p->theta)+dist_x(gen);
    p->y+=velocity*delta_t*sin(p->theta)+dist_x(gen);
    p->theta=limit_range(p->theta+yaw_rate*delta_t+dist_theta(gen));
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.
  for(vector<LandmarkObs>::iterator obs = observations.begin();obs!=observations.end();++obs) {
    double dmin=MAXFLOAT;
    for(vector<LandmarkObs>::iterator pred = predicted.begin();pred!=predicted.end();++pred) {
      double d=dist(pred->x,pred->y,obs->x,obs->y);
      if (d<dmin) {
        dmin=d;
        obs->id=pred->id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  
  vector<LandmarkObs> predicted;
  //main loop, going over all particles
  for (vector<Particle>::iterator p = particles.begin() ; p != particles.end(); ++p) {
    p->sense_x.clear();
    p->sense_y.clear();
    //calculate the predicted observations
    predicted.clear();
    //calculate all relevate predicted observations for this landmark
    for(vector<Map::single_landmark_s>::const_iterator lm= map_landmarks.landmark_list.begin();lm!=map_landmarks.landmark_list.end();++lm) {
      double d=dist(p->x, p->y, lm->x_f, lm->y_f);
      if(d<sensor_range){
        LandmarkObs obs;
        obs.id=lm->id_i;
        
        obs.x=(lm->x_f-p->x)*cos(p->theta)+(lm->y_f-p->y)*sin(p->theta);
        obs.y=-(lm->x_f-p->x)*sin(p->theta)+(lm->y_f-p->y)*cos(p->theta);
        predicted.push_back(obs);
      }
    }
    vector<LandmarkObs> p_observations= vector<LandmarkObs>(observations);
    //link the observations to the predicted observations
    dataAssociation(predicted, p_observations);
    double norm = 1/(2*M_PI*std_landmark[0]*std_landmark[1]);
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;
    //store all associated observations to the particle and update the weight based on the error
    p->weight=1;
    for (vector<LandmarkObs>::iterator obs=p_observations.begin();obs!=p_observations.end();++obs) {
      associations.push_back(obs->id);
      double x=obs->x*cos(p->theta)-obs->y*sin(p->theta)+p->x;
      double y=obs->x*sin(p->theta)+obs->y*cos(p->theta)+p->y;
      sense_x.push_back(x);
      sense_y.push_back(y);
      // find back landmark. This seems inefficient
      for(vector<Map::single_landmark_s>::const_iterator lm= map_landmarks.landmark_list.begin();lm!=map_landmarks.landmark_list.end();++lm) {
        if (lm->id_i==obs->id) {
          p->weight *= norm * exp(-(lm->x_f-x)*(lm->x_f-x)/(2*std_landmark[0]*std_landmark[0]) - (lm->y_f-y)*(lm->y_f-y)/(2*std_landmark[1]*std_landmark[1]));
          break;
        }
      }
    }
    SetAssociations(*p, associations, sense_x, sense_y);

  }
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html
}

void ParticleFilter::resample() {
  vector<double> weights;
  weights.reserve(num_particles);
  vector<Particle> newParticles;
  newParticles.reserve(num_particles);
  for (vector<Particle>::iterator p = particles.begin() ; p != particles.end(); ++p) {
    weights.push_back(p->weight);
  }
  discrete_distribution<double> dist(weights.begin(),weights.end());
  default_random_engine gen;
  for (unsigned int i=0;i<num_particles;++i) {
    int index=dist(gen);
    Particle p=particles[index];
    newParticles.push_back(p);
  }
  particles=newParticles;

  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  
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
