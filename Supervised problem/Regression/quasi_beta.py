# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 11:24:16 2017

@author: Futami
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.variational_inference import VariationalInference
from edward.models import RandomVariable
from edward.util import copy

try:
  from edward.models import Normal
  from tensorflow.contrib.distributions import kl_divergence
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))


def reduce_logmeanexp(input_tensor, axis=None, keep_dims=False):
  logsumexp = tf.reduce_logsumexp(input_tensor, axis, keep_dims)
  #logsumexpの中に最大値引く安定化は既に入っているのでtf.clipは書かなくて良い。
  input_tensor = tf.convert_to_tensor(input_tensor)
  n = input_tensor.shape.as_list()
  if axis is None:
    n = tf.cast(tf.reduce_prod(n), logsumexp.dtype)
  else:
    n = tf.cast(tf.reduce_prod(n[axis]), logsumexp.dtype)  
  return -tf.log(n) + logsumexp

class KLqp_beta(VariationalInference):

  def __init__(self,*args, **kwargs):
    super(KLqp_beta, self).__init__(*args, **kwargs)

  def initialize(self,n_samples=1,alpha=0.5,size=310,tot=310,kl_scaling=None, *args, **kwargs):
    if kl_scaling is None:
      kl_scaling = {}
    #self.importance=importance
    self.alpha=alpha
    self.size=size
    self.tot=tot
    self.n_samples = n_samples
    self.kl_scaling = kl_scaling
    return super(KLqp_beta, self).initialize(*args, **kwargs)

  def build_loss_and_gradients(self, var_list):
      
      cof = tf.constant(self.alpha,tf.float32)
      cof2 = tf.constant(self.alpha+1,tf.float32)
      M= tf.constant(self.size,tf.float32)
      N= tf.constant(self.tot,tf.float32)
      
      p_log_prob = [0.0] * self.n_samples
      q_log_prob = [0.0] * self.n_samples
      base_scope = tf.get_default_graph().unique_name("inference") + '/'
      for s in range(self.n_samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
        scope = base_scope + tf.get_default_graph().unique_name("sample")
        dict_swap = {}
        for x, qx in six.iteritems(self.data):
            if isinstance(x, RandomVariable):
                if isinstance(qx, RandomVariable):
                    qx_copy = copy(qx, scope=scope)
                    dict_swap[x] = qx_copy.value()
                else:
                    dict_swap[x] = qx

        for z, qz in six.iteritems(self.latent_vars):
          # Copy q(z) to obtain new set of posterior samples.
          qz_copy = copy(qz, scope=scope)
          dict_swap[z] = qz_copy.value()
          q_log_prob[s] += tf.reduce_sum(
              self.scale.get(z, 1.0) * qz_copy.log_prob(dict_swap[z]))
    
        for z in six.iterkeys(self.latent_vars):
          z_copy = copy(z, dict_swap, scope=scope)
          q_log_prob[s] -= tf.reduce_sum(
              self.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))
    
        for x in six.iterkeys(self.data):
          if isinstance(x, RandomVariable):
            x_copy = copy(x, dict_swap, scope=scope)
            p_log_prob[s] +=cof2/cof* tf.reduce_sum(
                self.scale.get(x, 1.0) *tf.exp( x_copy.log_prob(dict_swap[x])*cof))#-self.scale.get(x, 1.0) *1/cof2*(2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*reduce_logmeanexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-tf.log(N))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            
      kl_penalty = tf.reduce_sum([
          self.kl_scaling.get(z, 1.0) * tf.reduce_sum(kl_divergence(qz, z))
          for z, qz in six.iteritems(self.latent_vars)])
      #p_log_prob = tf.reduce_mean(p_log_prob)
      p_log_prob = tf.reduce_mean(p_log_prob)
      #p_log_prob = tf.reduce_mean(1/cof2*tf.reduce_logsumexp(p_log_prob*cof)+1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(1/cof2*300*reduce_logmeanexp(p_log_prob*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(tf.exp(p_log_prob*cof)*tf.exp(cof/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)))/cof
      q_log_prob = tf.reduce_mean(q_log_prob)
      
      if self.logging:
        tf.summary.scalar("loss/p_log_prob", p_log_prob,
                          collections=[self._summary_key])
        tf.summary.scalar("loss/q_log_prob", q_log_prob,
                          collections=[self._summary_key])
    
      #loss = -(tf.log(p_log_prob) - q_log_prob)
      loss = -(p_log_prob - q_log_prob)#- kl_penalty)
      grads = tf.gradients(loss, var_list)
      #print(grads)#tf.nn.l2_normalize(grads, 0, epsilon=1e-12, name=None)
      grads_and_vars = list(zip(grads, var_list))
      return loss, grads_and_vars

  def Influence_function(self, n_samples,var_list,Gamma=True):
      samples = n_samples
      cof = tf.constant(self.alpha,tf.float32)
      cof2 = tf.constant(self.alpha+1,tf.float32)
      M= tf.constant(self.size,tf.float32)
      N= tf.constant(self.tot,tf.float32)
      
      p_log_prob = [0.0] * samples
      q_log_prob = [0.0] * samples
      base_scope = tf.get_default_graph().unique_name("inference") + '/'
      for s in range(samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
        scope = base_scope + tf.get_default_graph().unique_name("sample")
        dict_swap = {}
        for x, qx in six.iteritems(self.data):
            if isinstance(x, RandomVariable):
                if isinstance(qx, RandomVariable):
                    qx_copy = copy(qx, scope=scope)
                    dict_swap[x] = qx_copy.value()
                else:
                    dict_swap[x] = qx

        for z, qz in six.iteritems(self.latent_vars):
          # Copy q(z) to obtain new set of posterior samples.
          qz_copy = copy(qz, scope=scope)
          dict_swap[z] = qz_copy.value()
          q_log_prob[s] += tf.reduce_sum(
              self.scale.get(z, 1.0) * qz_copy.log_prob(dict_swap[z]))
    
        for z in six.iterkeys(self.latent_vars):
          z_copy = copy(z, dict_swap, scope=scope)
          q_log_prob[s] -= tf.reduce_sum(
              self.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))
    
        for x in six.iterkeys(self.data):
          if isinstance(x, RandomVariable):
            x_copy = copy(x, dict_swap, scope=scope)
            #p_log_prob[s] += 1/cof*reduce_logmeanexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-tf.log(N))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(x_copy.log_prob(dict_swap[x])*cof)-tf.log(M))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            
            if Gamma:
                    p_log_prob[s] += tf.reduce_sum(
                            self.scale.get(x, 1.0) *tf.exp( x_copy.log_prob(dict_swap[x])*cof))
            else:
                    p_log_prob[s] += tf.reduce_sum(
                            self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))#*tf.exp(-cof/cof2*tf.log((2*3.1415*x.scale)**(cof/2)*(1+cof)*0.5))    
      kl_penalty = tf.reduce_sum([
          self.kl_scaling.get(z, 1.0) * tf.reduce_sum(kl_divergence(qz, z))
          for z, qz in six.iteritems(self.latent_vars)])
      #p_log_prob = tf.reduce_mean(p_log_prob)
      p_log_prob = tf.reduce_mean(p_log_prob)
      #p_log_prob = tf.reduce_mean(1/cof2*tf.reduce_logsumexp(p_log_prob*cof)+1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(1/cof2*300*reduce_logmeanexp(p_log_prob*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(tf.exp(p_log_prob*cof)*tf.exp(cof/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)))/cof
      q_log_prob = tf.reduce_mean(q_log_prob)
      
      if self.logging:
        tf.summary.scalar("loss/p_log_prob", p_log_prob,
                          collections=[self._summary_key])
        tf.summary.scalar("loss/q_log_prob", q_log_prob,
                          collections=[self._summary_key])
    
      #loss = -(tf.log(p_log_prob) - q_log_prob)
      loss = -(p_log_prob)# - q_log_prob)#- kl_penalty)
      grads = tf.gradients(loss, var_list)
      #print(grads)#tf.nn.l2_normalize(grads, 0, epsilon=1e-12, name=None)
      grads_and_vars = list(zip(grads, var_list))
      return loss, grads_and_vars
  
class KLqp_beta_binary(VariationalInference):

  def __init__(self,*args, **kwargs):
    super(KLqp_beta_binary, self).__init__(*args, **kwargs)

  def initialize(self,n_samples=1,alpha=0.5,size=310,tot=310,kl_scaling=None, *args, **kwargs):
    if kl_scaling is None:
      kl_scaling = {}
    #self.importance=importance
    self.alpha=alpha
    self.size=size
    self.tot=tot
    self.n_samples = n_samples
    self.kl_scaling = kl_scaling
    return super(KLqp_beta_binary, self).initialize(*args, **kwargs)

  def build_loss_and_gradients(self, var_list):
      
      cof = tf.constant(self.alpha,tf.float32)
      cof2 = tf.constant(self.alpha+1,tf.float32)
      M= tf.constant(self.size,tf.float32)
      N= tf.constant(self.tot,tf.float32)
      
      p_log_prob = [0.0] * self.n_samples
      q_log_prob = [0.0] * self.n_samples
      base_scope = tf.get_default_graph().unique_name("inference") + '/'
      for s in range(self.n_samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
        scope = base_scope + tf.get_default_graph().unique_name("sample")
        dict_swap = {}
        for x, qx in six.iteritems(self.data):
            if isinstance(x, RandomVariable):
                if isinstance(qx, RandomVariable):
                    qx_copy = copy(qx, scope=scope)
                    dict_swap[x] = qx_copy.value()
                else:
                    dict_swap[x] = qx

        for z, qz in six.iteritems(self.latent_vars):
          # Copy q(z) to obtain new set of posterior samples.
          qz_copy = copy(qz, scope=scope)
          dict_swap[z] = qz_copy.value()
          q_log_prob[s] += tf.reduce_sum(
              self.scale.get(z, 1.0) * qz_copy.log_prob(dict_swap[z]))
    
        for z in six.iterkeys(self.latent_vars):
          z_copy = copy(z, dict_swap, scope=scope)
          q_log_prob[s] -= tf.reduce_sum(
              self.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))
    
        for x in six.iterkeys(self.data):
          if isinstance(x, RandomVariable):
            x_copy = copy(x, dict_swap, scope=scope)
            p_log_prob[s] += cof2/cof*tf.reduce_sum(
                tf.exp( x_copy.log_prob(dict_swap[x])*cof))*N/M-tf.exp(tf.reduce_logsumexp(tf.log((x_copy.mean())**cof2+(1-x_copy.mean())**cof2)))*N/M
            
      kl_penalty = tf.reduce_sum([
          self.kl_scaling.get(z, 1.0) * tf.reduce_sum(kl_divergence(qz, z))
          for z, qz in six.iteritems(self.latent_vars)])
      #p_log_prob = tf.reduce_mean(p_log_prob)
      p_log_prob = tf.reduce_mean(p_log_prob)
      #p_log_prob = tf.reduce_mean(1/cof2*tf.reduce_logsumexp(p_log_prob*cof)+1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(1/cof2*300*reduce_logmeanexp(p_log_prob*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(tf.exp(p_log_prob*cof)*tf.exp(cof/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)))/cof
      q_log_prob = tf.reduce_mean(q_log_prob)
      
      if self.logging:
        tf.summary.scalar("loss/p_log_prob", p_log_prob,
                          collections=[self._summary_key])
        tf.summary.scalar("loss/q_log_prob", q_log_prob,
                          collections=[self._summary_key])
    
      #loss = -(tf.log(p_log_prob) - q_log_prob)
      loss = -(p_log_prob - q_log_prob)#- kl_penalty)
      grads = tf.gradients(loss, var_list)
      #print(grads)#tf.nn.l2_normalize(grads, 0, epsilon=1e-12, name=None)
      grads_and_vars = list(zip(grads, var_list))
      return loss, grads_and_vars
  
  def GRAD(self, n_samples,var_list,Gamma=True):
      samples = n_samples
      cof = tf.constant(self.alpha,tf.float32)
      cof2 = tf.constant(self.alpha+1,tf.float32)
      M= tf.constant(self.size,tf.float32)
      N= tf.constant(self.tot,tf.float32)
      
      p_log_prob = [0.0] * samples
      q_log_prob = [0.0] * samples
      base_scope = tf.get_default_graph().unique_name("inference") + '/'
      for s in range(samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
        scope = base_scope + tf.get_default_graph().unique_name("sample")
        dict_swap = {}
        for x, qx in six.iteritems(self.data):
            if isinstance(x, RandomVariable):
                if isinstance(qx, RandomVariable):
                    qx_copy = copy(qx, scope=scope)
                    dict_swap[x] = qx_copy.value()
                else:
                    dict_swap[x] = qx

        for z, qz in six.iteritems(self.latent_vars):
          # Copy q(z) to obtain new set of posterior samples.
          qz_copy = copy(qz, scope=scope)
          dict_swap[z] = qz_copy.value()
          q_log_prob[s] += tf.reduce_sum(
              self.scale.get(z, 1.0) * qz_copy.log_prob(dict_swap[z]))
    
        for z in six.iterkeys(self.latent_vars):
          z_copy = copy(z, dict_swap, scope=scope)
          q_log_prob[s] -= tf.reduce_sum(
              self.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))
    
        for x in six.iterkeys(self.data):
          if isinstance(x, RandomVariable):
            x_copy = copy(x, dict_swap, scope=scope)
            #p_log_prob[s] += 1/cof*reduce_logmeanexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-tf.log(N))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(x_copy.log_prob(dict_swap[x])*cof)-tf.log(M))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            
            if Gamma:
                    p_log_prob[s] += cof2/cof*tf.reduce_sum(
                        tf.exp( x_copy.log_prob(dict_swap[x])*cof))*N/M-tf.exp(tf.reduce_logsumexp(tf.log((x_copy.mean())**cof2+(1-x_copy.mean())**cof2)))*N/M
            else:
                    p_log_prob[s] += tf.reduce_sum(
                            self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))#*tf.exp(-cof/cof2*tf.log((2*3.1415*x.scale)**(cof/2)*(1+cof)*0.5))    
      kl_penalty = tf.reduce_sum([
          self.kl_scaling.get(z, 1.0) * tf.reduce_sum(kl_divergence(qz, z))
          for z, qz in six.iteritems(self.latent_vars)])
      #p_log_prob = tf.reduce_mean(p_log_prob)
      p_log_prob = tf.reduce_mean(p_log_prob)
      #p_log_prob = tf.reduce_mean(1/cof2*tf.reduce_logsumexp(p_log_prob*cof)+1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(1/cof2*300*reduce_logmeanexp(p_log_prob*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(tf.exp(p_log_prob*cof)*tf.exp(cof/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)))/cof
      q_log_prob = tf.reduce_mean(q_log_prob)
      
      if self.logging:
        tf.summary.scalar("loss/p_log_prob", p_log_prob,
                          collections=[self._summary_key])
        tf.summary.scalar("loss/q_log_prob", q_log_prob,
                          collections=[self._summary_key])
    
      #loss = -(tf.log(p_log_prob) - q_log_prob)
      #loss = -(p_log_prob - q_log_prob)#- kl_penalty)
      #grads = tf.gradients(loss, var_list)
      #print(grads)#tf.nn.l2_normalize(grads, 0, epsilon=1e-12, name=None)
      
      #H=tf.hessians(loss,var_list)
      #h2=tf.gradients(grads,var_list)
      #return_grads = [tf.gradients(grad,xs)[0] for grad in grads for xs in var_list]
      
  
      loss2 = -(p_log_prob)# - q_log_prob)#- kl_penalty)
      grads2 = tf.gradients(loss2, var_list)
      #OK=tf.matmul(tf.matrix_inverse(H),grads)

      return grads2
  
  def Hessian_ultimate(self, n_samples,var_list,vector,Gamma=True):
      samples = n_samples
      cof = tf.constant(self.alpha,tf.float32)
      cof2 = tf.constant(self.alpha+1,tf.float32)
      M= tf.constant(self.size,tf.float32)
      N= tf.constant(self.tot,tf.float32)
      
      p_log_prob = [0.0] * samples
      q_log_prob = [0.0] * samples
      base_scope = tf.get_default_graph().unique_name("inference") + '/'
      for s in range(samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
        scope = base_scope + tf.get_default_graph().unique_name("sample")
        dict_swap = {}
        for x, qx in six.iteritems(self.data):
            if isinstance(x, RandomVariable):
                if isinstance(qx, RandomVariable):
                    qx_copy = copy(qx, scope=scope)
                    dict_swap[x] = qx_copy.value()
                else:
                    dict_swap[x] = qx

        for z, qz in six.iteritems(self.latent_vars):
          # Copy q(z) to obtain new set of posterior samples.
          qz_copy = copy(qz, scope=scope)
          dict_swap[z] = qz_copy.value()
          q_log_prob[s] += tf.reduce_sum(
              self.scale.get(z, 1.0) * qz_copy.log_prob(dict_swap[z]))
    
        for z in six.iterkeys(self.latent_vars):
          z_copy = copy(z, dict_swap, scope=scope)
          q_log_prob[s] -= tf.reduce_sum(
              self.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))
    
        for x in six.iterkeys(self.data):
          if isinstance(x, RandomVariable):
            x_copy = copy(x, dict_swap, scope=scope)
            #p_log_prob[s] += 1/cof*reduce_logmeanexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-tf.log(N))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(x_copy.log_prob(dict_swap[x])*cof)-tf.log(M))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            
            if Gamma:
                    p_log_prob[s] += cof2/cof*tf.reduce_sum(
                        tf.exp( x_copy.log_prob(dict_swap[x])*cof))*N/M-tf.exp(tf.reduce_logsumexp(tf.log((x_copy.mean())**cof2+(1-x_copy.mean())**cof2)))*N/M
            else:
                    p_log_prob[s] += tf.reduce_sum(
                            self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))#*tf.exp(-cof/cof2*tf.log((2*3.1415*x.scale)**(cof/2)*(1+cof)*0.5))    
      kl_penalty = tf.reduce_sum([
          self.kl_scaling.get(z, 1.0) * tf.reduce_sum(kl_divergence(qz, z))
          for z, qz in six.iteritems(self.latent_vars)])
      #p_log_prob = tf.reduce_mean(p_log_prob)
      p_log_prob = tf.reduce_mean(p_log_prob)
      #p_log_prob = tf.reduce_mean(1/cof2*tf.reduce_logsumexp(p_log_prob*cof)+1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(1/cof2*300*reduce_logmeanexp(p_log_prob*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(tf.exp(p_log_prob*cof)*tf.exp(cof/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)))/cof
      q_log_prob = tf.reduce_mean(q_log_prob)
      
      if self.logging:
        tf.summary.scalar("loss/p_log_prob", p_log_prob,
                          collections=[self._summary_key])
        tf.summary.scalar("loss/q_log_prob", q_log_prob,
                          collections=[self._summary_key])
    
      #loss = -(tf.log(p_log_prob) - q_log_prob)
      loss = -(p_log_prob - q_log_prob)#- kl_penalty)
      grads = tf.gradients(loss, var_list)
      #print(grads)#tf.nn.l2_normalize(grads, 0, epsilon=1e-12, name=None)
      from tensorflow.python.ops import math_ops
      
      elemwise_products = [math_ops.multiply(grad_elem,vec) for grad_elem, vec in zip(grads, vector) if grad_elem is not None]
      #H=tf.hessians(loss,var_list)
      grads_with_none = tf.gradients(elemwise_products, var_list)
      return_grads = [grad_elem if grad_elem is not None else tf.zeros_like(x) for x, grad_elem in zip(var_list, grads_with_none)]
      
      #return_grads = [tf.gradients(grad,xs)[0] for grad in grads for xs in var_list]
      
  
      #loss2 = -(p_log_prob)# - q_log_prob)#- kl_penalty)
      #grads2 = tf.gradients(loss2, var_list)
      #OK=tf.matmul(tf.matrix_inverse(H),grads)

      return return_grads
  
class KLqp_KL(VariationalInference):

  def __init__(self,*args, **kwargs):
    super(KLqp_KL, self).__init__(*args, **kwargs)

  def initialize(self,n_samples=1,alpha=0.5,size=310,tot=310,kl_scaling=None, *args, **kwargs):
    if kl_scaling is None:
      kl_scaling = {}
    #self.importance=importance
    self.alpha=alpha
    self.size=size
    self.tot=tot
    self.n_samples = n_samples
    self.kl_scaling = kl_scaling
    return super(KLqp_KL, self).initialize(*args, **kwargs)

  def build_loss_and_gradients(self, var_list):
      
      cof = tf.constant(self.alpha,tf.float32)
      cof2 = tf.constant(self.alpha+1,tf.float32)
      M= tf.constant(self.size,tf.float32)
      N= tf.constant(self.tot,tf.float32)
      
      p_log_prob = [0.0] * self.n_samples
      q_log_prob = [0.0] * self.n_samples
      base_scope = tf.get_default_graph().unique_name("inference") + '/'
      for s in range(self.n_samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
        scope = base_scope + tf.get_default_graph().unique_name("sample")
        dict_swap = {}
        for x, qx in six.iteritems(self.data):
            if isinstance(x, RandomVariable):
                if isinstance(qx, RandomVariable):
                    qx_copy = copy(qx, scope=scope)
                    dict_swap[x] = qx_copy.value()
                else:
                    dict_swap[x] = qx

        for z, qz in six.iteritems(self.latent_vars):
          # Copy q(z) to obtain new set of posterior samples.
          qz_copy = copy(qz, scope=scope)
          dict_swap[z] = qz_copy.value()
          q_log_prob[s] += tf.reduce_sum(
              self.scale.get(z, 1.0) * qz_copy.log_prob(dict_swap[z]))
    
        for z in six.iterkeys(self.latent_vars):
          z_copy = copy(z, dict_swap, scope=scope)
          q_log_prob[s] -= tf.reduce_sum(
              self.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))
    
        for x in six.iterkeys(self.data):
          if isinstance(x, RandomVariable):
            x_copy = copy(x, dict_swap, scope=scope)
            #print(x.shape.as_list())
            #p_log_prob[s] += tf.reduce_sum(
            p_log_prob[s] += tf.reduce_sum(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))
            
      kl_penalty = tf.reduce_sum([
          self.kl_scaling.get(z, 1.0) * tf.reduce_sum(kl_divergence(qz, z))
          for z, qz in six.iteritems(self.latent_vars)])
      #p_log_prob = tf.reduce_mean(p_log_prob)
      p_log_prob = tf.reduce_mean(p_log_prob)
      #p_log_prob = tf.reduce_mean(1/cof2*tf.reduce_logsumexp(p_log_prob*cof)+1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(1/cof2*300*reduce_logmeanexp(p_log_prob*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(tf.exp(p_log_prob*cof)*tf.exp(cof/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)))/cof
      q_log_prob = tf.reduce_mean(q_log_prob)
      
      if self.logging:
        tf.summary.scalar("loss/p_log_prob", p_log_prob,
                          collections=[tf.get_default_graph().unique_name("summaries")])
        tf.summary.scalar("loss/q_log_prob", q_log_prob,
                          collections=[tf.get_default_graph().unique_name("summaries")])
    
      #loss = -(tf.log(p_log_prob) - q_log_prob)
      loss = -(p_log_prob*N - q_log_prob)#- kl_penalty)
      grads = tf.gradients(loss, var_list)
      #print(grads)#tf.nn.l2_normalize(grads, 0, epsilon=1e-12, name=None)
      grads_and_vars = list(zip(grads, var_list))
      return loss, grads_and_vars
  
  def Influence_function(self, n_samples,var_list,Gamma=True):
      samples = n_samples
      cof = tf.constant(self.alpha,tf.float32)
      cof2 = tf.constant(self.alpha+1,tf.float32)
      M= tf.constant(self.size,tf.float32)
      N= tf.constant(self.tot,tf.float32)
      
      p_log_prob = [0.0] * samples
      q_log_prob = [0.0] * samples
      base_scope = tf.get_default_graph().unique_name("inference") + '/'
      for s in range(samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
        scope = base_scope + tf.get_default_graph().unique_name("sample")
        dict_swap = {}
        for x, qx in six.iteritems(self.data):
            if isinstance(x, RandomVariable):
                if isinstance(qx, RandomVariable):
                    qx_copy = copy(qx, scope=scope)
                    dict_swap[x] = qx_copy.value()
                else:
                    dict_swap[x] = qx

        for z, qz in six.iteritems(self.latent_vars):
          # Copy q(z) to obtain new set of posterior samples.
          qz_copy = copy(qz, scope=scope)
          dict_swap[z] = qz_copy.value()
          q_log_prob[s] += tf.reduce_sum(
              self.scale.get(z, 1.0) * qz_copy.log_prob(dict_swap[z]))
    
        for z in six.iterkeys(self.latent_vars):
          z_copy = copy(z, dict_swap, scope=scope)
          q_log_prob[s] -= tf.reduce_sum(
              self.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))
    
        for x in six.iterkeys(self.data):
          if isinstance(x, RandomVariable):
            x_copy = copy(x, dict_swap, scope=scope)
            #p_log_prob[s] += 1/cof*reduce_logmeanexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-tf.log(N))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(x_copy.log_prob(dict_swap[x])*cof)-tf.log(M))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            
            if Gamma:
                    p_log_prob[s] +=tf.reduce_sum(self.scale.get(x, 1.0) *tf.exp(x_copy.log_prob(dict_swap[x])*cof)/((x_copy.mean())**cof2+(1-x_copy.mean())**cof2))
            else:
                    p_log_prob[s] += tf.reduce_sum(
                            self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))#*tf.exp(-cof/cof2*tf.log((2*3.1415*x.scale)**(cof/2)*(1+cof)*0.5))    
      kl_penalty = tf.reduce_sum([
          self.kl_scaling.get(z, 1.0) * tf.reduce_sum(kl_divergence(qz, z))
          for z, qz in six.iteritems(self.latent_vars)])
      #p_log_prob = tf.reduce_mean(p_log_prob)
      p_log_prob = tf.reduce_mean(p_log_prob)
      #p_log_prob = tf.reduce_mean(1/cof2*tf.reduce_logsumexp(p_log_prob*cof)+1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(1/cof2*300*reduce_logmeanexp(p_log_prob*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(tf.exp(p_log_prob*cof)*tf.exp(cof/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)))/cof
      q_log_prob = tf.reduce_mean(q_log_prob)
      
      if self.logging:
        tf.summary.scalar("loss/p_log_prob", p_log_prob,
                          collections=[self._summary_key])
        tf.summary.scalar("loss/q_log_prob", q_log_prob,
                          collections=[self._summary_key])
    
      #loss = -(tf.log(p_log_prob) - q_log_prob)
      loss = -(p_log_prob)# - q_log_prob)#- kl_penalty)
      grads = tf.gradients(loss, var_list)
      #print(grads)#tf.nn.l2_normalize(grads, 0, epsilon=1e-12, name=None)
      grads_and_vars = list(zip(grads, var_list))
      return loss, grads_and_vars
  
  def Hessian(self, n_samples,var_list,Gamma=True):
      samples = n_samples
      cof = tf.constant(self.alpha,tf.float32)
      cof2 = tf.constant(self.alpha+1,tf.float32)
      M= tf.constant(self.size,tf.float32)
      N= tf.constant(self.tot,tf.float32)
      
      p_log_prob = [0.0] * samples
      q_log_prob = [0.0] * samples
      base_scope = tf.get_default_graph().unique_name("inference") + '/'
      for s in range(samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
        scope = base_scope + tf.get_default_graph().unique_name("sample")
        dict_swap = {}
        for x, qx in six.iteritems(self.data):
            if isinstance(x, RandomVariable):
                if isinstance(qx, RandomVariable):
                    qx_copy = copy(qx, scope=scope)
                    dict_swap[x] = qx_copy.value()
                else:
                    dict_swap[x] = qx

        for z, qz in six.iteritems(self.latent_vars):
          # Copy q(z) to obtain new set of posterior samples.
          qz_copy = copy(qz, scope=scope)
          dict_swap[z] = qz_copy.value()
          q_log_prob[s] += tf.reduce_sum(
              self.scale.get(z, 1.0) * qz_copy.log_prob(dict_swap[z]))
    
        for z in six.iterkeys(self.latent_vars):
          z_copy = copy(z, dict_swap, scope=scope)
          q_log_prob[s] -= tf.reduce_sum(
              self.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))
    
        for x in six.iterkeys(self.data):
          if isinstance(x, RandomVariable):
            x_copy = copy(x, dict_swap, scope=scope)
            #p_log_prob[s] += 1/cof*reduce_logmeanexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-tf.log(N))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(x_copy.log_prob(dict_swap[x])*cof)-tf.log(M))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            
            if Gamma:
                    p_log_prob[s] +=tf.reduce_sum(self.scale.get(x, 1.0) *tf.exp(x_copy.log_prob(dict_swap[x])*cof)/((x_copy.mean())**cof2+(1-x_copy.mean())**cof2))
            else:
                    p_log_prob[s] += tf.reduce_sum(
                            self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))#*tf.exp(-cof/cof2*tf.log((2*3.1415*x.scale)**(cof/2)*(1+cof)*0.5))    
      kl_penalty = tf.reduce_sum([
          self.kl_scaling.get(z, 1.0) * tf.reduce_sum(kl_divergence(qz, z))
          for z, qz in six.iteritems(self.latent_vars)])
      #p_log_prob = tf.reduce_mean(p_log_prob)
      p_log_prob = tf.reduce_mean(p_log_prob)
      #p_log_prob = tf.reduce_mean(1/cof2*tf.reduce_logsumexp(p_log_prob*cof)+1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(1/cof2*300*reduce_logmeanexp(p_log_prob*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(tf.exp(p_log_prob*cof)*tf.exp(cof/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)))/cof
      q_log_prob = tf.reduce_mean(q_log_prob)
      
      if self.logging:
        tf.summary.scalar("loss/p_log_prob", p_log_prob,
                          collections=[self._summary_key])
        tf.summary.scalar("loss/q_log_prob", q_log_prob,
                          collections=[self._summary_key])
    
      #loss = -(tf.log(p_log_prob) - q_log_prob)
      loss = -(p_log_prob - q_log_prob)#- kl_penalty)
      grads = tf.gradients(loss, var_list)
      #print(grads)#tf.nn.l2_normalize(grads, 0, epsilon=1e-12, name=None)
      
      #H=tf.hessians(loss,var_list)
      #h2=tf.gradients(grads,var_list)
      #return_grads = [tf.gradients(grad,xs)[0] for grad in grads for xs in var_list]
      return_grads=[]
      n=len(grads)
      for i in np.arange(n):
          list2=[]
          for xs in var_list[i:]:
              list2.append(tf.gradients(grads[i],xs)[0])
          return_grads.append(list2)
      
      for i in np.arange(1,n):
          for j in range(i):
              return_grads[i].insert(j,return_grads[0][j])
      
      for i in np.arange(n):
          for j in np.arange(n):
              if return_grads[i][j]==None:
                  return_grads[i][j]=tf.constant(0)
    
      #return_grads = [tf.constant(0) if t == None else t  for t in con for con in return_grads]
      
      #loss2 = -(p_log_prob)# - q_log_prob)#- kl_penalty)
      #grads2 = tf.gradients(loss2, var_list)
      #OK=tf.matmul(tf.matrix_inverse(H),grads)

      return return_grads

  def Hessian2(self, n_samples,var_list,Gamma=True):
      samples = n_samples
      cof = tf.constant(self.alpha,tf.float32)
      cof2 = tf.constant(self.alpha+1,tf.float32)
      M= tf.constant(self.size,tf.float32)
      N= tf.constant(self.tot,tf.float32)
      
      p_log_prob = [0.0] * samples
      q_log_prob = [0.0] * samples
      base_scope = tf.get_default_graph().unique_name("inference") + '/'
      for s in range(samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
        scope = base_scope + tf.get_default_graph().unique_name("sample")
        dict_swap = {}
        for x, qx in six.iteritems(self.data):
            if isinstance(x, RandomVariable):
                if isinstance(qx, RandomVariable):
                    qx_copy = copy(qx, scope=scope)
                    dict_swap[x] = qx_copy.value()
                else:
                    dict_swap[x] = qx

        for z, qz in six.iteritems(self.latent_vars):
          # Copy q(z) to obtain new set of posterior samples.
          qz_copy = copy(qz, scope=scope)
          dict_swap[z] = qz_copy.value()
          q_log_prob[s] += tf.reduce_sum(
              self.scale.get(z, 1.0) * qz_copy.log_prob(dict_swap[z]))
    
        for z in six.iterkeys(self.latent_vars):
          z_copy = copy(z, dict_swap, scope=scope)
          q_log_prob[s] -= tf.reduce_sum(
              self.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))
    
        for x in six.iterkeys(self.data):
          if isinstance(x, RandomVariable):
            x_copy = copy(x, dict_swap, scope=scope)
            #p_log_prob[s] += 1/cof*reduce_logmeanexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-tf.log(N))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(x_copy.log_prob(dict_swap[x])*cof)-tf.log(M))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            
            if Gamma:
                    p_log_prob[s] +=tf.reduce_sum(self.scale.get(x, 1.0) *tf.exp(x_copy.log_prob(dict_swap[x])*cof)/((x_copy.mean())**cof2+(1-x_copy.mean())**cof2))
            else:
                    p_log_prob[s] += tf.reduce_sum(
                            self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))#*tf.exp(-cof/cof2*tf.log((2*3.1415*x.scale)**(cof/2)*(1+cof)*0.5))    
      kl_penalty = tf.reduce_sum([
          self.kl_scaling.get(z, 1.0) * tf.reduce_sum(kl_divergence(qz, z))
          for z, qz in six.iteritems(self.latent_vars)])
      #p_log_prob = tf.reduce_mean(p_log_prob)
      p_log_prob = tf.reduce_mean(p_log_prob)
      #p_log_prob = tf.reduce_mean(1/cof2*tf.reduce_logsumexp(p_log_prob*cof)+1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(1/cof2*300*reduce_logmeanexp(p_log_prob*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(tf.exp(p_log_prob*cof)*tf.exp(cof/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)))/cof
      q_log_prob = tf.reduce_mean(q_log_prob)
      
      if self.logging:
        tf.summary.scalar("loss/p_log_prob", p_log_prob,
                          collections=[self._summary_key])
        tf.summary.scalar("loss/q_log_prob", q_log_prob,
                          collections=[self._summary_key])
    
      #loss = -(tf.log(p_log_prob) - q_log_prob)
      loss = -(p_log_prob - q_log_prob)#- kl_penalty)
      grads = tf.gradients(loss, var_list)
      #print(grads)#tf.nn.l2_normalize(grads, 0, epsilon=1e-12, name=None)
      
      #H=tf.hessians(loss,var_list)
      #h2=tf.gradients(grads,var_list)
      return_grads = [tf.gradients(grad,xs)[0] for grad in grads for xs in var_list]
      
    
      return_grads = [tf.constant(0) if t == None else t  for t in return_grads]
      
      #loss2 = -(p_log_prob)# - q_log_prob)#- kl_penalty)
      #grads2 = tf.gradients(loss2, var_list)
      #OK=tf.matmul(tf.matrix_inverse(H),grads)

      return return_grads  

  def Hessian3(self, n_samples,var_list,vector,Gamma=True):
      samples = n_samples
      cof = tf.constant(self.alpha,tf.float32)
      cof2 = tf.constant(self.alpha+1,tf.float32)
      M= tf.constant(self.size,tf.float32)
      N= tf.constant(self.tot,tf.float32)
      
      p_log_prob = [0.0] * samples
      q_log_prob = [0.0] * samples
      base_scope = tf.get_default_graph().unique_name("inference") + '/'
      for s in range(samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
        scope = base_scope + tf.get_default_graph().unique_name("sample")
        dict_swap = {}
        for x, qx in six.iteritems(self.data):
            if isinstance(x, RandomVariable):
                if isinstance(qx, RandomVariable):
                    qx_copy = copy(qx, scope=scope)
                    dict_swap[x] = qx_copy.value()
                else:
                    dict_swap[x] = qx

        for z, qz in six.iteritems(self.latent_vars):
          # Copy q(z) to obtain new set of posterior samples.
          qz_copy = copy(qz, scope=scope)
          dict_swap[z] = qz_copy.value()
          q_log_prob[s] += tf.reduce_sum(
              self.scale.get(z, 1.0) * qz_copy.log_prob(dict_swap[z]))
    
        for z in six.iterkeys(self.latent_vars):
          z_copy = copy(z, dict_swap, scope=scope)
          q_log_prob[s] -= tf.reduce_sum(
              self.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))
    
        for x in six.iterkeys(self.data):
          if isinstance(x, RandomVariable):
            x_copy = copy(x, dict_swap, scope=scope)
            #p_log_prob[s] += 1/cof*reduce_logmeanexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-tf.log(N))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(x_copy.log_prob(dict_swap[x])*cof)-tf.log(M))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            
            if Gamma:
                    p_log_prob[s] += tf.reduce_sum(
                            self.scale.get(x, 1.0) *tf.exp( x_copy.log_prob(dict_swap[x])*cof))
            else:
                    p_log_prob[s] += tf.reduce_sum(
                            self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))#*tf.exp(-cof/cof2*tf.log((2*3.1415*x.scale)**(cof/2)*(1+cof)*0.5))    
      kl_penalty = tf.reduce_sum([
          self.kl_scaling.get(z, 1.0) * tf.reduce_sum(kl_divergence(qz, z))
          for z, qz in six.iteritems(self.latent_vars)])
      #p_log_prob = tf.reduce_mean(p_log_prob)
      p_log_prob = tf.reduce_mean(p_log_prob)
      #p_log_prob = tf.reduce_mean(1/cof2*tf.reduce_logsumexp(p_log_prob*cof)+1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(1/cof2*300*reduce_logmeanexp(p_log_prob*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(tf.exp(p_log_prob*cof)*tf.exp(cof/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)))/cof
      q_log_prob = tf.reduce_mean(q_log_prob)
      
      if self.logging:
        tf.summary.scalar("loss/p_log_prob", p_log_prob,
                          collections=[self._summary_key])
        tf.summary.scalar("loss/q_log_prob", q_log_prob,
                          collections=[self._summary_key])
    
      #loss = -(tf.log(p_log_prob) - q_log_prob)
      loss = -(p_log_prob - q_log_prob)#- kl_penalty)
      grads = tf.gradients(loss, var_list)
      #print(grads)#tf.nn.l2_normalize(grads, 0, epsilon=1e-12, name=None)
      from tensorflow.python.ops import math_ops
      elemwise_products = [grad_elem*vector[i] for grad_elem, i in zip(grads, np.arange(len(var_list))) if grad_elem is not None]
      #H=tf.hessians(loss,var_list)
      grads_with_none = tf.gradients(elemwise_products, var_list)
      return_grads = [grad_elem if grad_elem is not None else tf.zeros_like(x) for x, grad_elem in zip(var_list, grads_with_none)]
      
      #return_grads = [tf.gradients(grad,xs)[0] for grad in grads for xs in var_list]
      
  
      #loss2 = -(p_log_prob)# - q_log_prob)#- kl_penalty)
      #grads2 = tf.gradients(loss2, var_list)
      #OK=tf.matmul(tf.matrix_inverse(H),grads)

      return return_grads

  def GRAD(self, n_samples,var_list,Gamma=True):
      samples = n_samples
      cof = tf.constant(self.alpha,tf.float32)
      cof2 = tf.constant(self.alpha+1,tf.float32)
      M= tf.constant(self.size,tf.float32)
      N= tf.constant(self.tot,tf.float32)
      
      p_log_prob = [0.0] * samples
      q_log_prob = [0.0] * samples
      base_scope = tf.get_default_graph().unique_name("inference") + '/'
      for s in range(samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
        scope = base_scope + tf.get_default_graph().unique_name("sample")
        dict_swap = {}
        for x, qx in six.iteritems(self.data):
            if isinstance(x, RandomVariable):
                if isinstance(qx, RandomVariable):
                    qx_copy = copy(qx, scope=scope)
                    dict_swap[x] = qx_copy.value()
                else:
                    dict_swap[x] = qx

        for z, qz in six.iteritems(self.latent_vars):
          # Copy q(z) to obtain new set of posterior samples.
          qz_copy = copy(qz, scope=scope)
          dict_swap[z] = qz_copy.value()
          q_log_prob[s] += tf.reduce_sum(
              self.scale.get(z, 1.0) * qz_copy.log_prob(dict_swap[z]))
    
        for z in six.iterkeys(self.latent_vars):
          z_copy = copy(z, dict_swap, scope=scope)
          q_log_prob[s] -= tf.reduce_sum(
              self.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))
    
        for x in six.iterkeys(self.data):
          if isinstance(x, RandomVariable):
            x_copy = copy(x, dict_swap, scope=scope)
            #p_log_prob[s] += 1/cof*reduce_logmeanexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-tf.log(N))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(x_copy.log_prob(dict_swap[x])*cof)-tf.log(M))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            
            if Gamma:
                    p_log_prob[s] +=cof2/cof*tf.reduce_sum(self.scale.get(x, 1.0) *tf.exp(x_copy.log_prob(dict_swap[x])*cof)/((x_copy.mean())**cof2+(1-x_copy.mean())**cof2))
            else:
                    p_log_prob[s] += tf.reduce_sum(
                            self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))#*tf.exp(-cof/cof2*tf.log((2*3.1415*x.scale)**(cof/2)*(1+cof)*0.5))    
      kl_penalty = tf.reduce_sum([
          self.kl_scaling.get(z, 1.0) * tf.reduce_sum(kl_divergence(qz, z))
          for z, qz in six.iteritems(self.latent_vars)])
      #p_log_prob = tf.reduce_mean(p_log_prob)
      p_log_prob = tf.reduce_mean(p_log_prob)
      #p_log_prob = tf.reduce_mean(1/cof2*tf.reduce_logsumexp(p_log_prob*cof)+1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(1/cof2*300*reduce_logmeanexp(p_log_prob*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(tf.exp(p_log_prob*cof)*tf.exp(cof/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)))/cof
      q_log_prob = tf.reduce_mean(q_log_prob)
      
      if self.logging:
        tf.summary.scalar("loss/p_log_prob", p_log_prob,
                          collections=[self._summary_key])
        tf.summary.scalar("loss/q_log_prob", q_log_prob,
                          collections=[self._summary_key])
    
      #loss = -(tf.log(p_log_prob) - q_log_prob)
      #loss = -(p_log_prob - q_log_prob)#- kl_penalty)
      #grads = tf.gradients(loss, var_list)
      #print(grads)#tf.nn.l2_normalize(grads, 0, epsilon=1e-12, name=None)
      
      #H=tf.hessians(loss,var_list)
      #h2=tf.gradients(grads,var_list)
      #return_grads = [tf.gradients(grad,xs)[0] for grad in grads for xs in var_list]
      
  
      loss2 = -(p_log_prob)# - q_log_prob)#- kl_penalty)
      grads2 = tf.gradients(loss2, var_list)
      #OK=tf.matmul(tf.matrix_inverse(H),grads)

      return grads2
  
  def Hessian_ultimate(self, n_samples,var_list,vector,Gamma=True):
      samples = n_samples
      cof = tf.constant(self.alpha,tf.float32)
      cof2 = tf.constant(self.alpha+1,tf.float32)
      M= tf.constant(self.size,tf.float32)
      N= tf.constant(self.tot,tf.float32)
      
      p_log_prob = [0.0] * samples
      q_log_prob = [0.0] * samples
      base_scope = tf.get_default_graph().unique_name("inference") + '/'
      for s in range(samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
        scope = base_scope + tf.get_default_graph().unique_name("sample")
        dict_swap = {}
        for x, qx in six.iteritems(self.data):
            if isinstance(x, RandomVariable):
                if isinstance(qx, RandomVariable):
                    qx_copy = copy(qx, scope=scope)
                    dict_swap[x] = qx_copy.value()
                else:
                    dict_swap[x] = qx

        for z, qz in six.iteritems(self.latent_vars):
          # Copy q(z) to obtain new set of posterior samples.
          qz_copy = copy(qz, scope=scope)
          dict_swap[z] = qz_copy.value()
          q_log_prob[s] += tf.reduce_sum(
              self.scale.get(z, 1.0) * qz_copy.log_prob(dict_swap[z]))
    
        for z in six.iterkeys(self.latent_vars):
          z_copy = copy(z, dict_swap, scope=scope)
          q_log_prob[s] -= tf.reduce_sum(
              self.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))
    
        for x in six.iterkeys(self.data):
          if isinstance(x, RandomVariable):
            x_copy = copy(x, dict_swap, scope=scope)
            #p_log_prob[s] += 1/cof*reduce_logmeanexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-tf.log(N))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(x_copy.log_prob(dict_swap[x])*cof)-tf.log(M))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            
            if Gamma:
                    p_log_prob[s] +=cof2/cof*tf.reduce_sum(self.scale.get(x, 1.0) *tf.exp(x_copy.log_prob(dict_swap[x])*cof)/((x_copy.mean())**cof2+(1-x_copy.mean())**cof2))
            else:
                    p_log_prob[s] += tf.reduce_sum(
                            self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))#*tf.exp(-cof/cof2*tf.log((2*3.1415*x.scale)**(cof/2)*(1+cof)*0.5))    
      kl_penalty = tf.reduce_sum([
          self.kl_scaling.get(z, 1.0) * tf.reduce_sum(kl_divergence(qz, z))
          for z, qz in six.iteritems(self.latent_vars)])
      #p_log_prob = tf.reduce_mean(p_log_prob)
      p_log_prob = tf.reduce_mean(p_log_prob)
      #p_log_prob = tf.reduce_mean(1/cof2*tf.reduce_logsumexp(p_log_prob*cof)+1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(1/cof2*300*reduce_logmeanexp(p_log_prob*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(tf.exp(p_log_prob*cof)*tf.exp(cof/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)))/cof
      q_log_prob = tf.reduce_mean(q_log_prob)
      
      if self.logging:
        tf.summary.scalar("loss/p_log_prob", p_log_prob,
                          collections=[self._summary_key])
        tf.summary.scalar("loss/q_log_prob", q_log_prob,
                          collections=[self._summary_key])
    
      #loss = -(tf.log(p_log_prob) - q_log_prob)
      loss = -(p_log_prob - q_log_prob)#- kl_penalty)
      grads = tf.gradients(loss, var_list)
      #print(grads)#tf.nn.l2_normalize(grads, 0, epsilon=1e-12, name=None)
      from tensorflow.python.ops import math_ops
      
      elemwise_products = [math_ops.multiply(grad_elem,vec) for grad_elem, vec in zip(grads, vector) if grad_elem is not None]
      #H=tf.hessians(loss,var_list)
      grads_with_none = tf.gradients(elemwise_products, var_list)
      return_grads = [grad_elem if grad_elem is not None else tf.zeros_like(x) for x, grad_elem in zip(var_list, grads_with_none)]
      
      #return_grads = [tf.gradients(grad,xs)[0] for grad in grads for xs in var_list]
      
  
      #loss2 = -(p_log_prob)# - q_log_prob)#- kl_penalty)
      #grads2 = tf.gradients(loss2, var_list)
      #OK=tf.matmul(tf.matrix_inverse(H),grads)

      return return_grads
  
  def GRAD2(self, n_samples,var_list,Gamma=True):
      samples = n_samples
      cof = tf.constant(self.alpha,tf.float32)
      cof2 = tf.constant(self.alpha+1,tf.float32)
      M= tf.constant(self.size,tf.float32)
      N= tf.constant(self.tot,tf.float32)
      
      p_log_prob = [0.0] * samples
      q_log_prob = [0.0] * samples
      base_scope = tf.get_default_graph().unique_name("inference") + '/'
      for s in range(samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
        scope = base_scope + tf.get_default_graph().unique_name("sample")
        dict_swap = {}
        for x, qx in six.iteritems(self.data):
            if isinstance(x, RandomVariable):
                if isinstance(qx, RandomVariable):
                    qx_copy = copy(qx, scope=scope)
                    dict_swap[x] = qx_copy.value()
                else:
                    dict_swap[x] = qx

        for z, qz in six.iteritems(self.latent_vars):
          # Copy q(z) to obtain new set of posterior samples.
          qz_copy = copy(qz, scope=scope)
          dict_swap[z] = qz_copy.value()
          q_log_prob[s] += tf.reduce_sum(
              self.scale.get(z, 1.0) * qz_copy.log_prob(dict_swap[z]))
    
        for z in six.iterkeys(self.latent_vars):
          z_copy = copy(z, dict_swap, scope=scope)
          q_log_prob[s] -= tf.reduce_sum(
              self.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))
    
        for x in six.iterkeys(self.data):
          if isinstance(x, RandomVariable):
            x_copy = copy(x, dict_swap, scope=scope)
            #p_log_prob[s] += 1/cof*reduce_logmeanexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-tf.log(N))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(x_copy.log_prob(dict_swap[x])*cof)-tf.log(M))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)

            p_log_prob[s] += tf.reduce_sum(
                            self.scale.get(x, 1.0) * x_copy.prob(dict_swap[x]))#*tf.exp(-cof/cof2*tf.log((2*3.1415*x.scale)**(cof/2)*(1+cof)*0.5))    
      kl_penalty = tf.reduce_sum([
          self.kl_scaling.get(z, 1.0) * tf.reduce_sum(kl_divergence(qz, z))
          for z, qz in six.iteritems(self.latent_vars)])
      #p_log_prob = tf.reduce_mean(p_log_prob)
      p_log_prob = tf.reduce_mean(p_log_prob)
      #p_log_prob = tf.reduce_mean(1/cof2*tf.reduce_logsumexp(p_log_prob*cof)+1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(1/cof2*300*reduce_logmeanexp(p_log_prob*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(tf.exp(p_log_prob*cof)*tf.exp(cof/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)))/cof
      q_log_prob = tf.reduce_mean(q_log_prob)
      
      if self.logging:
        tf.summary.scalar("loss/p_log_prob", p_log_prob,
                          collections=[self._summary_key])
        tf.summary.scalar("loss/q_log_prob", q_log_prob,
                          collections=[self._summary_key])
    
      #loss = -(tf.log(p_log_prob) - q_log_prob)
      #loss = -(p_log_prob - q_log_prob)#- kl_penalty)
      #grads = tf.gradients(loss, var_list)
      #print(grads)#tf.nn.l2_normalize(grads, 0, epsilon=1e-12, name=None)
      
      #H=tf.hessians(loss,var_list)
      #h2=tf.gradients(grads,var_list)
      #return_grads = [tf.gradients(grad,xs)[0] for grad in grads for xs in var_list]
      
  
      loss2 = -(p_log_prob)# - q_log_prob)#- kl_penalty)
      grads2 = tf.gradients(loss2, var_list)
      return_grads = [grad_elem if grad_elem is not None else tf.zeros_like(x) for x, grad_elem in zip(var_list, grads2)]
      
      #return_grads
      #OK=tf.matmul(tf.matrix_inverse(H),grads)

      return return_grads
  
class KLqp_gamma_binary(VariationalInference):

  def __init__(self,*args, **kwargs):
    super(KLqp_gamma_binary, self).__init__(*args, **kwargs)

  def initialize(self,n_samples=1,alpha=0.5,size=310,tot=310,kl_scaling=None, *args, **kwargs):
    if kl_scaling is None:
      kl_scaling = {}
    #self.importance=importance
    self.alpha=alpha
    self.size=size
    self.tot=tot
    self.n_samples = n_samples
    self.kl_scaling = kl_scaling
    return super(KLqp_gamma_binary, self).initialize(*args, **kwargs)

  def build_loss_and_gradients(self, var_list):
      
      cof = tf.constant(self.alpha,tf.float32)
      cof2 = tf.constant(self.alpha+1,tf.float32)
      M= tf.constant(self.size,tf.float32)
      N= tf.constant(self.tot,tf.float32)
      
      p_log_prob = [0.0] * self.n_samples
      q_log_prob = [0.0] * self.n_samples
      base_scope = tf.get_default_graph().unique_name("inference") + '/'
      for s in range(self.n_samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
        scope = base_scope + tf.get_default_graph().unique_name("sample")
        dict_swap = {}
        for x, qx in six.iteritems(self.data):
            if isinstance(x, RandomVariable):
                if isinstance(qx, RandomVariable):
                    qx_copy = copy(qx, scope=scope)
                    dict_swap[x] = qx_copy.value()
                else:
                    dict_swap[x] = qx

        for z, qz in six.iteritems(self.latent_vars):
          # Copy q(z) to obtain new set of posterior samples.
          qz_copy = copy(qz, scope=scope)
          dict_swap[z] = qz_copy.value()
          q_log_prob[s] += tf.reduce_sum(
              self.scale.get(z, 1.0) * qz_copy.log_prob(dict_swap[z]))
    
        for z in six.iterkeys(self.latent_vars):
          z_copy = copy(z, dict_swap, scope=scope)
          q_log_prob[s] -= tf.reduce_sum(
              self.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))
    
        for x in six.iterkeys(self.data):
          if isinstance(x, RandomVariable):
            x_copy = copy(x, dict_swap, scope=scope)
            #print(x.shape.as_list())
            #p_log_prob[s] += tf.reduce_sum(
            #    self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))#*tf.exp(-cof/cof2*tf.log((2*3.1415*x.scale)**(cof/2)*(1+cof)*0.5))
            #p_log_prob[s] += 1/cof*reduce_logmeanexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-tf.log(N))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            p_log_prob[s] +=N *cof2/cof*tf.reduce_mean(tf.exp(x_copy.log_prob(dict_swap[x])*cof)/((x_copy.mean())**cof2+(1-x_copy.mean())**cof2)**(cof/cof2))
            
      kl_penalty = tf.reduce_sum([
          self.kl_scaling.get(z, 1.0) * tf.reduce_sum(kl_divergence(qz, z))
          for z, qz in six.iteritems(self.latent_vars)])
      #p_log_prob = tf.reduce_mean(p_log_prob)
      p_log_prob = tf.reduce_mean(p_log_prob)
      #p_log_prob = tf.reduce_mean(1/cof2*tf.reduce_logsumexp(p_log_prob*cof)+1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(1/cof2*300*reduce_logmeanexp(p_log_prob*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(tf.exp(p_log_prob*cof)*tf.exp(cof/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)))/cof
      q_log_prob = tf.reduce_mean(q_log_prob)
      
      if self.logging:
        tf.summary.scalar("loss/p_log_prob", p_log_prob,
                          collections=[tf.get_default_graph().unique_name("summaries")])
        tf.summary.scalar("loss/q_log_prob", q_log_prob,
                          collections=[tf.get_default_graph().unique_name("summaries")])
    
      #loss = -(tf.log(p_log_prob) - q_log_prob)
      loss = -(p_log_prob - q_log_prob)#- kl_penalty)
      grads = tf.gradients(loss, var_list)
      #print(grads)#tf.nn.l2_normalize(grads, 0, epsilon=1e-12, name=None)
      grads_and_vars = list(zip(grads, var_list))
      return loss, grads_and_vars
  
  def Influence_function(self, n_samples,var_list,Gamma=True):
      samples = n_samples
      cof = tf.constant(self.alpha,tf.float32)
      cof2 = tf.constant(self.alpha+1,tf.float32)
      M= tf.constant(self.size,tf.float32)
      N= tf.constant(self.tot,tf.float32)
      
      p_log_prob = [0.0] * samples
      q_log_prob = [0.0] * samples
      base_scope = tf.get_default_graph().unique_name("inference") + '/'
      for s in range(samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
        scope = base_scope + tf.get_default_graph().unique_name("sample")
        dict_swap = {}
        for x, qx in six.iteritems(self.data):
            if isinstance(x, RandomVariable):
                if isinstance(qx, RandomVariable):
                    qx_copy = copy(qx, scope=scope)
                    dict_swap[x] = qx_copy.value()
                else:
                    dict_swap[x] = qx

        for z, qz in six.iteritems(self.latent_vars):
          # Copy q(z) to obtain new set of posterior samples.
          qz_copy = copy(qz, scope=scope)
          dict_swap[z] = qz_copy.value()
          q_log_prob[s] += tf.reduce_sum(
              self.scale.get(z, 1.0) * qz_copy.log_prob(dict_swap[z]))
    
        for z in six.iterkeys(self.latent_vars):
          z_copy = copy(z, dict_swap, scope=scope)
          q_log_prob[s] -= tf.reduce_sum(
              self.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))
    
        for x in six.iterkeys(self.data):
          if isinstance(x, RandomVariable):
            x_copy = copy(x, dict_swap, scope=scope)
            #p_log_prob[s] += 1/cof*reduce_logmeanexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-tf.log(N))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(x_copy.log_prob(dict_swap[x])*cof)-tf.log(M))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            
            if Gamma:
                    p_log_prob[s] +=tf.reduce_sum(self.scale.get(x, 1.0) *tf.exp(x_copy.log_prob(dict_swap[x])*cof)/((x_copy.mean())**cof2+(1-x_copy.mean())**cof2))
            else:
                    p_log_prob[s] += tf.reduce_sum(
                            self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))#*tf.exp(-cof/cof2*tf.log((2*3.1415*x.scale)**(cof/2)*(1+cof)*0.5))    
      kl_penalty = tf.reduce_sum([
          self.kl_scaling.get(z, 1.0) * tf.reduce_sum(kl_divergence(qz, z))
          for z, qz in six.iteritems(self.latent_vars)])
      #p_log_prob = tf.reduce_mean(p_log_prob)
      p_log_prob = tf.reduce_mean(p_log_prob)
      #p_log_prob = tf.reduce_mean(1/cof2*tf.reduce_logsumexp(p_log_prob*cof)+1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(1/cof2*300*reduce_logmeanexp(p_log_prob*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(tf.exp(p_log_prob*cof)*tf.exp(cof/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)))/cof
      q_log_prob = tf.reduce_mean(q_log_prob)
      
      if self.logging:
        tf.summary.scalar("loss/p_log_prob", p_log_prob,
                          collections=[self._summary_key])
        tf.summary.scalar("loss/q_log_prob", q_log_prob,
                          collections=[self._summary_key])
    
      #loss = -(tf.log(p_log_prob) - q_log_prob)
      loss = -(p_log_prob)# - q_log_prob)#- kl_penalty)
      grads = tf.gradients(loss, var_list)
      #print(grads)#tf.nn.l2_normalize(grads, 0, epsilon=1e-12, name=None)
      grads_and_vars = list(zip(grads, var_list))
      return loss, grads_and_vars
  
  def Hessian(self, n_samples,var_list,Gamma=True):
      samples = n_samples
      cof = tf.constant(self.alpha,tf.float32)
      cof2 = tf.constant(self.alpha+1,tf.float32)
      M= tf.constant(self.size,tf.float32)
      N= tf.constant(self.tot,tf.float32)
      
      p_log_prob = [0.0] * samples
      q_log_prob = [0.0] * samples
      base_scope = tf.get_default_graph().unique_name("inference") + '/'
      for s in range(samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
        scope = base_scope + tf.get_default_graph().unique_name("sample")
        dict_swap = {}
        for x, qx in six.iteritems(self.data):
            if isinstance(x, RandomVariable):
                if isinstance(qx, RandomVariable):
                    qx_copy = copy(qx, scope=scope)
                    dict_swap[x] = qx_copy.value()
                else:
                    dict_swap[x] = qx

        for z, qz in six.iteritems(self.latent_vars):
          # Copy q(z) to obtain new set of posterior samples.
          qz_copy = copy(qz, scope=scope)
          dict_swap[z] = qz_copy.value()
          q_log_prob[s] += tf.reduce_sum(
              self.scale.get(z, 1.0) * qz_copy.log_prob(dict_swap[z]))
    
        for z in six.iterkeys(self.latent_vars):
          z_copy = copy(z, dict_swap, scope=scope)
          q_log_prob[s] -= tf.reduce_sum(
              self.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))
    
        for x in six.iterkeys(self.data):
          if isinstance(x, RandomVariable):
            x_copy = copy(x, dict_swap, scope=scope)
            #p_log_prob[s] += 1/cof*reduce_logmeanexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-tf.log(N))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(x_copy.log_prob(dict_swap[x])*cof)-tf.log(M))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            
            if Gamma:
                    p_log_prob[s] +=tf.reduce_sum(self.scale.get(x, 1.0) *tf.exp(x_copy.log_prob(dict_swap[x])*cof)/((x_copy.mean())**cof2+(1-x_copy.mean())**cof2))
            else:
                    p_log_prob[s] += tf.reduce_sum(
                            self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))#*tf.exp(-cof/cof2*tf.log((2*3.1415*x.scale)**(cof/2)*(1+cof)*0.5))    
      kl_penalty = tf.reduce_sum([
          self.kl_scaling.get(z, 1.0) * tf.reduce_sum(kl_divergence(qz, z))
          for z, qz in six.iteritems(self.latent_vars)])
      #p_log_prob = tf.reduce_mean(p_log_prob)
      p_log_prob = tf.reduce_mean(p_log_prob)
      #p_log_prob = tf.reduce_mean(1/cof2*tf.reduce_logsumexp(p_log_prob*cof)+1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(1/cof2*300*reduce_logmeanexp(p_log_prob*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(tf.exp(p_log_prob*cof)*tf.exp(cof/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)))/cof
      q_log_prob = tf.reduce_mean(q_log_prob)
      
      if self.logging:
        tf.summary.scalar("loss/p_log_prob", p_log_prob,
                          collections=[self._summary_key])
        tf.summary.scalar("loss/q_log_prob", q_log_prob,
                          collections=[self._summary_key])
    
      #loss = -(tf.log(p_log_prob) - q_log_prob)
      loss = -(p_log_prob - q_log_prob)#- kl_penalty)
      grads = tf.gradients(loss, var_list)
      #print(grads)#tf.nn.l2_normalize(grads, 0, epsilon=1e-12, name=None)
      
      #H=tf.hessians(loss,var_list)
      #h2=tf.gradients(grads,var_list)
      #return_grads = [tf.gradients(grad,xs)[0] for grad in grads for xs in var_list]
      return_grads=[]
      n=len(grads)
      for i in np.arange(n):
          list2=[]
          for xs in var_list[i:]:
              list2.append(tf.gradients(grads[i],xs)[0])
          return_grads.append(list2)
      
      for i in np.arange(1,n):
          for j in range(i):
              return_grads[i].insert(j,return_grads[0][j])
      
      for i in np.arange(n):
          for j in np.arange(n):
              if return_grads[i][j]==None:
                  return_grads[i][j]=tf.constant(0)
    
      #return_grads = [tf.constant(0) if t == None else t  for t in con for con in return_grads]
      
      #loss2 = -(p_log_prob)# - q_log_prob)#- kl_penalty)
      #grads2 = tf.gradients(loss2, var_list)
      #OK=tf.matmul(tf.matrix_inverse(H),grads)

      return return_grads

  def Hessian2(self, n_samples,var_list,Gamma=True):
      samples = n_samples
      cof = tf.constant(self.alpha,tf.float32)
      cof2 = tf.constant(self.alpha+1,tf.float32)
      M= tf.constant(self.size,tf.float32)
      N= tf.constant(self.tot,tf.float32)
      
      p_log_prob = [0.0] * samples
      q_log_prob = [0.0] * samples
      base_scope = tf.get_default_graph().unique_name("inference") + '/'
      for s in range(samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
        scope = base_scope + tf.get_default_graph().unique_name("sample")
        dict_swap = {}
        for x, qx in six.iteritems(self.data):
            if isinstance(x, RandomVariable):
                if isinstance(qx, RandomVariable):
                    qx_copy = copy(qx, scope=scope)
                    dict_swap[x] = qx_copy.value()
                else:
                    dict_swap[x] = qx

        for z, qz in six.iteritems(self.latent_vars):
          # Copy q(z) to obtain new set of posterior samples.
          qz_copy = copy(qz, scope=scope)
          dict_swap[z] = qz_copy.value()
          q_log_prob[s] += tf.reduce_sum(
              self.scale.get(z, 1.0) * qz_copy.log_prob(dict_swap[z]))
    
        for z in six.iterkeys(self.latent_vars):
          z_copy = copy(z, dict_swap, scope=scope)
          q_log_prob[s] -= tf.reduce_sum(
              self.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))
    
        for x in six.iterkeys(self.data):
          if isinstance(x, RandomVariable):
            x_copy = copy(x, dict_swap, scope=scope)
            #p_log_prob[s] += 1/cof*reduce_logmeanexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-tf.log(N))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(x_copy.log_prob(dict_swap[x])*cof)-tf.log(M))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            
            if Gamma:
                    p_log_prob[s] +=tf.reduce_sum(self.scale.get(x, 1.0) *tf.exp(x_copy.log_prob(dict_swap[x])*cof)/((x_copy.mean())**cof2+(1-x_copy.mean())**cof2))
            else:
                    p_log_prob[s] += tf.reduce_sum(
                            self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))#*tf.exp(-cof/cof2*tf.log((2*3.1415*x.scale)**(cof/2)*(1+cof)*0.5))    
      kl_penalty = tf.reduce_sum([
          self.kl_scaling.get(z, 1.0) * tf.reduce_sum(kl_divergence(qz, z))
          for z, qz in six.iteritems(self.latent_vars)])
      #p_log_prob = tf.reduce_mean(p_log_prob)
      p_log_prob = tf.reduce_mean(p_log_prob)
      #p_log_prob = tf.reduce_mean(1/cof2*tf.reduce_logsumexp(p_log_prob*cof)+1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(1/cof2*300*reduce_logmeanexp(p_log_prob*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(tf.exp(p_log_prob*cof)*tf.exp(cof/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)))/cof
      q_log_prob = tf.reduce_mean(q_log_prob)
      
      if self.logging:
        tf.summary.scalar("loss/p_log_prob", p_log_prob,
                          collections=[self._summary_key])
        tf.summary.scalar("loss/q_log_prob", q_log_prob,
                          collections=[self._summary_key])
    
      #loss = -(tf.log(p_log_prob) - q_log_prob)
      loss = -(p_log_prob - q_log_prob)#- kl_penalty)
      grads = tf.gradients(loss, var_list)
      #print(grads)#tf.nn.l2_normalize(grads, 0, epsilon=1e-12, name=None)
      
      #H=tf.hessians(loss,var_list)
      #h2=tf.gradients(grads,var_list)
      return_grads = [tf.gradients(grad,xs)[0] for grad in grads for xs in var_list]
      
    
      return_grads = [tf.constant(0) if t == None else t  for t in return_grads]
      
      #loss2 = -(p_log_prob)# - q_log_prob)#- kl_penalty)
      #grads2 = tf.gradients(loss2, var_list)
      #OK=tf.matmul(tf.matrix_inverse(H),grads)

      return return_grads  

  def Hessian3(self, n_samples,var_list,vector,Gamma=True):
      samples = n_samples
      cof = tf.constant(self.alpha,tf.float32)
      cof2 = tf.constant(self.alpha+1,tf.float32)
      M= tf.constant(self.size,tf.float32)
      N= tf.constant(self.tot,tf.float32)
      
      p_log_prob = [0.0] * samples
      q_log_prob = [0.0] * samples
      base_scope = tf.get_default_graph().unique_name("inference") + '/'
      for s in range(samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
        scope = base_scope + tf.get_default_graph().unique_name("sample")
        dict_swap = {}
        for x, qx in six.iteritems(self.data):
            if isinstance(x, RandomVariable):
                if isinstance(qx, RandomVariable):
                    qx_copy = copy(qx, scope=scope)
                    dict_swap[x] = qx_copy.value()
                else:
                    dict_swap[x] = qx

        for z, qz in six.iteritems(self.latent_vars):
          # Copy q(z) to obtain new set of posterior samples.
          qz_copy = copy(qz, scope=scope)
          dict_swap[z] = qz_copy.value()
          q_log_prob[s] += tf.reduce_sum(
              self.scale.get(z, 1.0) * qz_copy.log_prob(dict_swap[z]))
    
        for z in six.iterkeys(self.latent_vars):
          z_copy = copy(z, dict_swap, scope=scope)
          q_log_prob[s] -= tf.reduce_sum(
              self.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))
    
        for x in six.iterkeys(self.data):
          if isinstance(x, RandomVariable):
            x_copy = copy(x, dict_swap, scope=scope)
            #p_log_prob[s] += 1/cof*reduce_logmeanexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-tf.log(N))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(x_copy.log_prob(dict_swap[x])*cof)-tf.log(M))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            
            if Gamma:
                    p_log_prob[s] += tf.reduce_sum(
                            self.scale.get(x, 1.0) *tf.exp( x_copy.log_prob(dict_swap[x])*cof))
            else:
                    p_log_prob[s] += tf.reduce_sum(
                            self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))#*tf.exp(-cof/cof2*tf.log((2*3.1415*x.scale)**(cof/2)*(1+cof)*0.5))    
      kl_penalty = tf.reduce_sum([
          self.kl_scaling.get(z, 1.0) * tf.reduce_sum(kl_divergence(qz, z))
          for z, qz in six.iteritems(self.latent_vars)])
      #p_log_prob = tf.reduce_mean(p_log_prob)
      p_log_prob = tf.reduce_mean(p_log_prob)
      #p_log_prob = tf.reduce_mean(1/cof2*tf.reduce_logsumexp(p_log_prob*cof)+1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(1/cof2*300*reduce_logmeanexp(p_log_prob*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(tf.exp(p_log_prob*cof)*tf.exp(cof/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)))/cof
      q_log_prob = tf.reduce_mean(q_log_prob)
      
      if self.logging:
        tf.summary.scalar("loss/p_log_prob", p_log_prob,
                          collections=[self._summary_key])
        tf.summary.scalar("loss/q_log_prob", q_log_prob,
                          collections=[self._summary_key])
    
      #loss = -(tf.log(p_log_prob) - q_log_prob)
      loss = -(p_log_prob - q_log_prob)#- kl_penalty)
      grads = tf.gradients(loss, var_list)
      #print(grads)#tf.nn.l2_normalize(grads, 0, epsilon=1e-12, name=None)
      from tensorflow.python.ops import math_ops
      elemwise_products = [grad_elem*vector[i] for grad_elem, i in zip(grads, np.arange(len(var_list))) if grad_elem is not None]
      #H=tf.hessians(loss,var_list)
      grads_with_none = tf.gradients(elemwise_products, var_list)
      return_grads = [grad_elem if grad_elem is not None else tf.zeros_like(x) for x, grad_elem in zip(var_list, grads_with_none)]
      
      #return_grads = [tf.gradients(grad,xs)[0] for grad in grads for xs in var_list]
      
  
      #loss2 = -(p_log_prob)# - q_log_prob)#- kl_penalty)
      #grads2 = tf.gradients(loss2, var_list)
      #OK=tf.matmul(tf.matrix_inverse(H),grads)

      return return_grads

  def GRAD(self, n_samples,var_list,Gamma=True):
      samples = n_samples
      cof = tf.constant(self.alpha,tf.float32)
      cof2 = tf.constant(self.alpha+1,tf.float32)
      M= tf.constant(self.size,tf.float32)
      N= tf.constant(self.tot,tf.float32)
      
      p_log_prob = [0.0] * samples
      q_log_prob = [0.0] * samples
      base_scope = tf.get_default_graph().unique_name("inference") + '/'
      for s in range(samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
        scope = base_scope + tf.get_default_graph().unique_name("sample")
        dict_swap = {}
        for x, qx in six.iteritems(self.data):
            if isinstance(x, RandomVariable):
                if isinstance(qx, RandomVariable):
                    qx_copy = copy(qx, scope=scope)
                    dict_swap[x] = qx_copy.value()
                else:
                    dict_swap[x] = qx

        for z, qz in six.iteritems(self.latent_vars):
          # Copy q(z) to obtain new set of posterior samples.
          qz_copy = copy(qz, scope=scope)
          dict_swap[z] = qz_copy.value()
          q_log_prob[s] += tf.reduce_sum(
              self.scale.get(z, 1.0) * qz_copy.log_prob(dict_swap[z]))
    
        for z in six.iterkeys(self.latent_vars):
          z_copy = copy(z, dict_swap, scope=scope)
          q_log_prob[s] -= tf.reduce_sum(
              self.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))
    
        for x in six.iterkeys(self.data):
          if isinstance(x, RandomVariable):
            x_copy = copy(x, dict_swap, scope=scope)
            #p_log_prob[s] += 1/cof*reduce_logmeanexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-tf.log(N))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(x_copy.log_prob(dict_swap[x])*cof)-tf.log(M))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            
            if Gamma:
                    p_log_prob[s] += cof2/cof*tf.reduce_sum(
                            tf.exp( x_copy.log_prob(dict_swap[x])*cof))*N/M-tf.exp(tf.reduce_logsumexp(tf.log((x_copy.mean())**cof2+(1-x_copy.mean())**cof2)))*N/M

            else:
                    p_log_prob[s] += tf.reduce_sum(
                            self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))#*tf.exp(-cof/cof2*tf.log((2*3.1415*x.scale)**(cof/2)*(1+cof)*0.5))    
      kl_penalty = tf.reduce_sum([
          self.kl_scaling.get(z, 1.0) * tf.reduce_sum(kl_divergence(qz, z))
          for z, qz in six.iteritems(self.latent_vars)])
      #p_log_prob = tf.reduce_mean(p_log_prob)
      p_log_prob = tf.reduce_mean(p_log_prob)
      #p_log_prob = tf.reduce_mean(1/cof2*tf.reduce_logsumexp(p_log_prob*cof)+1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(1/cof2*300*reduce_logmeanexp(p_log_prob*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(tf.exp(p_log_prob*cof)*tf.exp(cof/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)))/cof
      q_log_prob = tf.reduce_mean(q_log_prob)
      
      if self.logging:
        tf.summary.scalar("loss/p_log_prob", p_log_prob,
                          collections=[self._summary_key])
        tf.summary.scalar("loss/q_log_prob", q_log_prob,
                          collections=[self._summary_key])
    
      #loss = -(tf.log(p_log_prob) - q_log_prob)
      #loss = -(p_log_prob - q_log_prob)#- kl_penalty)
      #grads = tf.gradients(loss, var_list)
      #print(grads)#tf.nn.l2_normalize(grads, 0, epsilon=1e-12, name=None)
      
      #H=tf.hessians(loss,var_list)
      #h2=tf.gradients(grads,var_list)
      #return_grads = [tf.gradients(grad,xs)[0] for grad in grads for xs in var_list]
      
  
      loss2 = -(p_log_prob)# - q_log_prob)#- kl_penalty)
      grads2 = tf.gradients(loss2, var_list)
      #OK=tf.matmul(tf.matrix_inverse(H),grads)

      return grads2
  
  def Hessian_ultimate(self, n_samples,var_list,vector,Gamma=True):
      samples = n_samples
      cof = tf.constant(self.alpha,tf.float32)
      cof2 = tf.constant(self.alpha+1,tf.float32)
      M= tf.constant(self.size,tf.float32)
      N= tf.constant(self.tot,tf.float32)
      
      p_log_prob = [0.0] * samples
      q_log_prob = [0.0] * samples
      base_scope = tf.get_default_graph().unique_name("inference") + '/'
      for s in range(samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
        scope = base_scope + tf.get_default_graph().unique_name("sample")
        dict_swap = {}
        for x, qx in six.iteritems(self.data):
            if isinstance(x, RandomVariable):
                if isinstance(qx, RandomVariable):
                    qx_copy = copy(qx, scope=scope)
                    dict_swap[x] = qx_copy.value()
                else:
                    dict_swap[x] = qx

        for z, qz in six.iteritems(self.latent_vars):
          # Copy q(z) to obtain new set of posterior samples.
          qz_copy = copy(qz, scope=scope)
          dict_swap[z] = qz_copy.value()
          q_log_prob[s] += tf.reduce_sum(
              self.scale.get(z, 1.0) * qz_copy.log_prob(dict_swap[z]))
    
        for z in six.iterkeys(self.latent_vars):
          z_copy = copy(z, dict_swap, scope=scope)
          q_log_prob[s] -= tf.reduce_sum(
              self.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))
    
        for x in six.iterkeys(self.data):
          if isinstance(x, RandomVariable):
            x_copy = copy(x, dict_swap, scope=scope)
            #p_log_prob[s] += 1/cof*reduce_logmeanexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-tf.log(N))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(x_copy.log_prob(dict_swap[x])*cof)-tf.log(M))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            
            if Gamma:
                    p_log_prob[s] += cof2/cof*tf.reduce_sum(
                            tf.exp( x_copy.log_prob(dict_swap[x])*cof))*N/M-tf.exp(tf.reduce_logsumexp(tf.log((x_copy.mean())**cof2+(1-x_copy.mean())**cof2)))*N/M

            else:
                    p_log_prob[s] += tf.reduce_sum(
                            self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x]))#*tf.exp(-cof/cof2*tf.log((2*3.1415*x.scale)**(cof/2)*(1+cof)*0.5))    
      kl_penalty = tf.reduce_sum([
          self.kl_scaling.get(z, 1.0) * tf.reduce_sum(kl_divergence(qz, z))
          for z, qz in six.iteritems(self.latent_vars)])
      #p_log_prob = tf.reduce_mean(p_log_prob)
      p_log_prob = tf.reduce_mean(p_log_prob)
      #p_log_prob = tf.reduce_mean(1/cof2*tf.reduce_logsumexp(p_log_prob*cof)+1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(1/cof2*300*reduce_logmeanexp(p_log_prob*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(tf.exp(p_log_prob*cof)*tf.exp(cof/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)))/cof
      q_log_prob = tf.reduce_mean(q_log_prob)
      
      if self.logging:
        tf.summary.scalar("loss/p_log_prob", p_log_prob,
                          collections=[self._summary_key])
        tf.summary.scalar("loss/q_log_prob", q_log_prob,
                          collections=[self._summary_key])
    
      #loss = -(tf.log(p_log_prob) - q_log_prob)
      loss = -(p_log_prob - q_log_prob)#- kl_penalty)
      grads = tf.gradients(loss, var_list)
      #print(grads)#tf.nn.l2_normalize(grads, 0, epsilon=1e-12, name=None)
      from tensorflow.python.ops import math_ops
      
      elemwise_products = [math_ops.multiply(grad_elem,vec) for grad_elem, vec in zip(grads, vector) if grad_elem is not None]
      #H=tf.hessians(loss,var_list)
      grads_with_none = tf.gradients(elemwise_products, var_list)
      return_grads = [grad_elem if grad_elem is not None else tf.zeros_like(x) for x, grad_elem in zip(var_list, grads_with_none)]
      
      #return_grads = [tf.gradients(grad,xs)[0] for grad in grads for xs in var_list]
      
  
      #loss2 = -(p_log_prob)# - q_log_prob)#- kl_penalty)
      #grads2 = tf.gradients(loss2, var_list)
      #OK=tf.matmul(tf.matrix_inverse(H),grads)

      return return_grads
  
  def GRAD2(self, n_samples,var_list,Gamma=True):
      samples = n_samples
      cof = tf.constant(self.alpha,tf.float32)
      cof2 = tf.constant(self.alpha+1,tf.float32)
      M= tf.constant(self.size,tf.float32)
      N= tf.constant(self.tot,tf.float32)
      
      p_log_prob = [0.0] * samples
      q_log_prob = [0.0] * samples
      base_scope = tf.get_default_graph().unique_name("inference") + '/'
      for s in range(samples):
    # Form dictionary in order to replace conditioning on prior or
    # observed variable with conditioning on a specific value.
        scope = base_scope + tf.get_default_graph().unique_name("sample")
        dict_swap = {}
        for x, qx in six.iteritems(self.data):
            if isinstance(x, RandomVariable):
                if isinstance(qx, RandomVariable):
                    qx_copy = copy(qx, scope=scope)
                    dict_swap[x] = qx_copy.value()
                else:
                    dict_swap[x] = qx

        for z, qz in six.iteritems(self.latent_vars):
          # Copy q(z) to obtain new set of posterior samples.
          qz_copy = copy(qz, scope=scope)
          dict_swap[z] = qz_copy.value()
          q_log_prob[s] += tf.reduce_sum(
              self.scale.get(z, 1.0) * qz_copy.log_prob(dict_swap[z]))
    
        for z in six.iterkeys(self.latent_vars):
          z_copy = copy(z, dict_swap, scope=scope)
          q_log_prob[s] -= tf.reduce_sum(
              self.scale.get(z, 1.0) * z_copy.log_prob(dict_swap[z]))
    
        for x in six.iterkeys(self.data):
          if isinstance(x, RandomVariable):
            x_copy = copy(x, dict_swap, scope=scope)
            #p_log_prob[s] += 1/cof*reduce_logmeanexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(self.scale.get(x, 1.0) * x_copy.log_prob(dict_swap[x])*cof)-tf.log(N))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)
            #p_log_prob[s] += 1/cof*(tf.reduce_logsumexp(x_copy.log_prob(dict_swap[x])*cof)-tf.log(M))-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)

            p_log_prob[s] += tf.reduce_sum(
                            self.scale.get(x, 1.0) * x_copy.prob(dict_swap[x]))#*tf.exp(-cof/cof2*tf.log((2*3.1415*x.scale)**(cof/2)*(1+cof)*0.5))    
      kl_penalty = tf.reduce_sum([
          self.kl_scaling.get(z, 1.0) * tf.reduce_sum(kl_divergence(qz, z))
          for z, qz in six.iteritems(self.latent_vars)])
      #p_log_prob = tf.reduce_mean(p_log_prob)
      p_log_prob = tf.reduce_mean(p_log_prob)
      #p_log_prob = tf.reduce_mean(1/cof2*tf.reduce_logsumexp(p_log_prob*cof)+1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(1/cof2*300*reduce_logmeanexp(p_log_prob*cof)-1/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5))
      #p_log_prob = tf.reduce_mean(tf.exp(p_log_prob*cof)*tf.exp(cof/cof2*tf.log((2*3.1415*1)**(cof/2)*(1+cof)**0.5)))/cof
      q_log_prob = tf.reduce_mean(q_log_prob)
      
      if self.logging:
        tf.summary.scalar("loss/p_log_prob", p_log_prob,
                          collections=[self._summary_key])
        tf.summary.scalar("loss/q_log_prob", q_log_prob,
                          collections=[self._summary_key])
    
      #loss = -(tf.log(p_log_prob) - q_log_prob)
      #loss = -(p_log_prob - q_log_prob)#- kl_penalty)
      #grads = tf.gradients(loss, var_list)
      #print(grads)#tf.nn.l2_normalize(grads, 0, epsilon=1e-12, name=None)
      
      #H=tf.hessians(loss,var_list)
      #h2=tf.gradients(grads,var_list)
      #return_grads = [tf.gradients(grad,xs)[0] for grad in grads for xs in var_list]
      
  
      loss2 = -(p_log_prob)# - q_log_prob)#- kl_penalty)
      grads2 = tf.gradients(loss2, var_list)
      return_grads = [grad_elem if grad_elem is not None else tf.zeros_like(x) for x, grad_elem in zip(var_list, grads2)]
      
      #return_grads
      #OK=tf.matmul(tf.matrix_inverse(H),grads)

      return return_grads