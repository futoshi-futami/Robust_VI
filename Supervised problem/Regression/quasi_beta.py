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
                  # the above second term for the unbiasedness need not to be included in the objective function because  it will be constant when we consider the regression problem, and thus it will vanish when we take the gradient. 
      kl_penalty = tf.reduce_sum([
          self.kl_scaling.get(z, 1.0) * tf.reduce_sum(kl_divergence(qz, z))
          for z, qz in six.iteritems(self.latent_vars)])
      
      p_log_prob = tf.reduce_mean(p_log_prob)
      q_log_prob = tf.reduce_mean(q_log_prob)
      
      if self.logging:
        tf.summary.scalar("loss/p_log_prob", p_log_prob,
                          collections=[self._summary_key])
        tf.summary.scalar("loss/q_log_prob", q_log_prob,
                          collections=[self._summary_key])
      loss = -(p_log_prob - q_log_prob)
      grads = tf.gradients(loss, var_list)
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

      p_log_prob = tf.reduce_mean(p_log_prob)
      q_log_prob = tf.reduce_mean(q_log_prob)
      
      if self.logging:
        tf.summary.scalar("loss/p_log_prob", p_log_prob,
                          collections=[self._summary_key])
        tf.summary.scalar("loss/q_log_prob", q_log_prob,
                          collections=[self._summary_key])
    
      
      loss = -(p_log_prob - q_log_prob)
      grads = tf.gradients(loss, var_list)
      
      grads_and_vars = list(zip(grads, var_list))
      return loss, grads_and_vars
  