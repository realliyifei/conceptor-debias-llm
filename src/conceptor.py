# -*- coding: utf-8 -*-
""" 
This is code to calculate the conceptor, as well as boolean operators. Copied from conceptor_fxns.py
Also, another ref: https://github.com/duguyue100/conceptors/blob/master/conceptors/logic.py
"""
import numpy as np

def train_Conceptor(x, alpha = 1):
  print("starting...")
  #x = orig_embd.vectors
  print(x.shape)
  
  #Calculate the correlation matrix
  R = x.dot(x.T)/(x.shape[1])
  print("R calculated")
  
  #Calculate the conceptor matrix
  C = R @ (np.linalg.inv(R + alpha ** (-2) * np.eye(x.shape[0])))
  print("C calculated")
    
  return C, R

def NOT(C, out_mode = "simple"):
  """
  Compute NOT operation of conceptor.
  
  @param R: conceptor matrix
  @param out_mode: output mode ("simple"/"complete")
  
  @return not_C: NOT C
  @return U: eigen vectors of not_C
  @return S: eigen values of not_C
  """
  
  dim = C.shape[0]
  
  not_C = np.eye(dim) - C
  
  if out_mode == "complete":
    U, S, _ = np.linalg.svd(not_C)
    return not_C, U, S
  else:
    return not_C

def AND(C, B, out_mode = "simple", tol = 1e-14):
  """
  Compute AND Operation of two conceptor matrices
  
  @param C: a conceptor matrix
  @param B: another conceptor matrix
  @param out_mode: output mode ("simple"/"complete")
  @param tol: adjust parameter for almost zero
  
  @return C_and_B: C AND B
  @return U: eigen vectors of C_and_B
  @return S: eigen values of C_and_B
  """
  
  dim = C.shape[0]
  
  UC, SC, _ = np.linalg.svd(C)
  UB, SB, _ = np.linalg.svd(B)
  
  num_rank_C = np.sum((SC > tol).astype(float))
  num_rank_B = np.sum((SB > tol).astype(float))
  
  UC0 = UC[:, int(num_rank_C):]
  UB0 = UB[:, int(num_rank_B):]
  
  W, sigma, _ = np.linalg.svd(UC0.dot(UC0.T) + UB0.dot(UB0.T))
  num_rank_sigma = np.sum((sigma > tol).astype(float))
  Wgk = W[:, int(num_rank_sigma):]
  
  C_and_B = Wgk.dot(np.linalg.inv(Wgk.T.dot(np.linalg.pinv(C, tol) + np.linalg.pinv(B, tol) - np.eye(dim)).dot(Wgk))).dot(Wgk.T)

  if out_mode =="complete":
    U, S, _ = np.linalg.svd(C_and_B)
    return C_and_B, U, S
  else:
    return C_and_B

def OR(R, Q, out_mode="simple"):
  """
  Compute OR operation of two conceptors
  
  @param R: covariance matrix of a conceptor
  @param Q: covariance matrix of a conceptor
  @param out_mode: output mode ("simple"/"complete")
  
  @return not_R: negative of R
  @return U: eigen vectors of not_R
  @return S: eigen values of not_R
  """
  
  R_or_Q=NOT(AND(NOT(R), NOT(Q)))

  if out_mode=="simple":
    return R_or_Q;
  elif out_mode=="complete":
    U, S, _=np.linalg.svd(R_or_Q)
    return R_or_Q, U, S
  else:
    return R_or_Q

#Compute the conceptor matrix
def post_process_cn_matrix(x, alpha = 1):
  C, R = train_Conceptor(x, alpha=alpha)
  
  #Calculate the negation of the conceptor matrix
  negC = NOT(C)
  print("negC calculated")
  
  #Post-process the vocab matrix
  newX = (negC @ x).T
  print(newX.shape)
  return negC, newX, R