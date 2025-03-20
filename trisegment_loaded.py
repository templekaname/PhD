
# ACE model with improved mesh quality, with node insertion algorithm

# Import libraries
import numpy as np
from numpy.linalg import norm
pi = np.pi
import scipy.optimize as opt
import scipy.integrate as sint
import scipy.sparse as spar
import scipy.sparse.linalg as spla
from copy import deepcopy as dc
import functools as ft
from tqdm import tqdm
# Import visualiation libraries
import plotly as ply
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as io
io.renderers.default = "vscode"
from IPython.display import display, HTML

# workaround for displaying LaTeX symbols in plotly
ply.offline.init_notebook_mode()
display(HTML(
    '<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG"></script>'
))

# Class ACEModel
class trisegmentmodel:

  def __init__(self, ns_init, nt, T, strtype="constant"):
    # set up numerical parameters
    self.ns = ns_init # initial node size
    self.nt = nt # time step size
    self.T = T # final time
    self.dt = T/(nt-1) # time increment
    self.strtype = strtype # strain type used for priary growth zone
    # set up matrices used in the end for visualization
    self.ns_list = []
    self.C_list = []
    self.Ci_list = []
    self.A_list = []
    self.dCidt_passive_list = []
    self.dCidt_active_list = []
    self.dCidt_list = []
    self.l_list = []
    self.r_list = []
    self.s_list = []
    self.N_list = []
    self.M_list = []
    # put the first ns in list
    self.ns_list.append(dc(self.ns))
  
  def set_geometry(self, l0, lg, lt, ldot, r0, rdot, A0=pi/2):
    # set up geometry parameters
    self.l0 = l0 # initial length
    self.lg = lg # growth zone length
    self.lt = lt # transition zone length
    self.ldot = ldot # elongation speed
    self.r0 = r0 # initial radius
    self.rdot = rdot # radial increase speed
    # initialize lists
    self.l = np.ones(self.ns-1) * self.l0/(self.ns-1) # segment length
    self.s = np.insert(np.cumsum(self.l), 0, 0) # nodal coordinates
    mid = (self.s[1:] + self.s[:-1])/2 # midpoint coordinates
    self.r = self.r0 + self.rdot/self.ldot*(self.s[-1]-self.lg-mid) * np.heaviside(self.s[-1]-self.lg-mid,0) # radius distribution
    # set up maximum tolerance of segment length
    self.l_max = 3*self.l0/(self.ns-1)
    # initialize angle and curvature
    self.loaded = False
    self.A0 = A0
    self.A = A0*np.ones(self.ns)
    self.C = np.zeros(self.ns-1)
    self.Ci = np.zeros(self.ns-1) # set up initial intrinsic curvature
    # add to lists
    self.A_list.append(dc(self.A))
    self.Ci_list.append(dc(self.Ci))
    self.l_list.append(dc(self.l))
    self.r_list.append(dc(self.r))
    self.s_list.append(dc(self.s))

  def set_load(self, E, rho_g):
    self.rho_g = rho_g
    self.E = E
    self.loaded = True
    self.deform()

  def set_sensitivity(self, alpha, beta, gamma):
    # set up sensitivity parameters
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma

  def deform(self, max_itr=1000, tolerance=1E-7):

    coef = 4*self.rho_g/self.E

    sin_coef = np.append(np.cumsum((self.r**2*self.l)[::-1])[::-1],0)
    sin_coef[0] = 0
    sin_coef[-1] = 0

    B = np.zeros((self.ns, self.ns))
    b1 = np.zeros(self.ns)
    sins = np.zeros(self.ns)

    B[0,0] = 1
    B[-1,-2] = -1
    B[-1,-1] = 1

    b1[0] = self.A0
    b1[-1] = self.Ci[-1]*self.l[-1]

    for i in range(1,self.ns-1):
      B[i,i-1] = self.r[i-1]**4 / self.l[i-1]
      B[i,i] = - self.r[i-1]**4 / self.l[i-1] - self.r[i]**4 / self.l[i]
      B[i,i+1] = self.r[i]**4 / self.l[i]
      b1[i] = self.r[i]**4 * self.Ci[i] - self.r[i-1]**4 * self.Ci[i-1]

    B_sparse = spar.csc_matrix(B)

    A = dc(self.A)
    counter = 0
    while True:
      counter += 1
      sins = sin_coef*np.sin(A)
      coss = sin_coef*np.cos(A)
      J = B_sparse + spar.diags(coss)
      F = B_sparse@A - b1 + coef*sins
      if norm(F) > tolerance:
        A -= spla.spsolve(J, F)
      else:
        break
      if counter > max_itr:
        raise Exception("Does not converge !!!!")

    self.A = A
    self.C = (A[1:]-A[:-1])/self.l


  def insert_nodes(self):
    bad_index = np.argwhere(self.l > self.l_max)
    if bad_index.size != 0:
      s_index_list = np.hstack(bad_index)
      for s_index in s_index_list:
        self.l = np.insert(self.l, s_index, self.l[s_index]/2)
        self.l[s_index+1] /= 2
        self.r = np.insert(self.r, s_index, self.r[s_index])
        self.Ci = np.insert(self.Ci, s_index, self.Ci[s_index])
        self.ns += 1
        s_index_list += 1

  def Edotfunc(self, s, type="constant"):
    if type == "constant":
      return self.ldot/self.lg * np.heaviside(s - (self.s[-1]-self.lg),0)
    elif type == "quad":
      return 3*self.ldot/(2*self.lg) * (1 - 4/self.lg**2 * (self.s[-1] - s - self.lg/2)**2) * np.heaviside(s - (self.s[-1]-self.lg),0)
    elif type == "linear1":
      return 2*self.ldot/self.lg**2 * (self.s[-1] - s) * np.heaviside(s - (self.s[-1]-self.lg),0)
    elif type == "linear2":
      return 2*self.ldot/self.lg**2 * (self.lg - self.s[-1] + s) * np.heaviside(s - (self.s[-1]-self.lg),0)
    else:
      raise Exception("Please specify a valid parameter indicator")

  def iterate(self):
    mid = (self.s[1:]+self.s[:-1])/2
    theta_mid = (self.A[1:]+self.A[:-1])/2
    strrate = self.Edotfunc(mid, self.strtype)
    dCi_primary = strrate * (-self.beta / self.r0 * np.sin(theta_mid) - self.gamma*self.Ci) * self.dt
    # dCi_transition = self.alpha * self.rdot/self.r * (-self.gamma*self.Ci) * self.dt * (np.heaviside(mid-(self.l-self.lg-self.lt),0) - np.heaviside(mid-(self.l-self.lg),0))
    # dCi_secondary = self.alpha * self.rdot/self.r * (-self.beta / self.r * np.sin(theta_mid) - self.gamma*self.Ci) * self.dt * np.heaviside((self.l-self.lg-self.lt)-mid,0)
    dCi_secondary = self.alpha * self.rdot/self.r * (-self.beta / self.r * np.sin(theta_mid) - self.gamma*self.Ci) * self.dt * np.heaviside((self.s[-1]-self.lg-self.lt)-mid,0)
    dCi = dCi_primary + dCi_secondary
    self.Ci += dCi
    self.l *= 1 + strrate*self.dt
    self.r = self.r0 + self.rdot/self.ldot*(self.s[-1]-self.lg-mid) * np.heaviside(self.s[-1]-self.lg-mid,0)
    self.insert_nodes()

    self.s = np.insert(np.cumsum(self.l), 0, 0)
    self.A = self.A0 + np.insert(np.cumsum(self.Ci*self.l), 0, 0)

    self.ns_list.append(dc(self.ns))
    self.A_list.append(dc(self.A))
    self.s_list.append(dc(self.s))
    self.Ci_list.append(dc(self.Ci))
    self.l_list.append(dc(self.l))
    self.r_list.append(dc(self.r))
    self.dCidt_list.append(dc(dCi)/self.dt)

  def simulate(self, t_list=[], var_list=[]):
    t_list = np.sort(np.array((t_list)))
    t_index_list = np.floor(t_list/self.dt).astype(int)
    modify_index = 0
    for t in range(int(self.nt)):
      if t in t_index_list:
        setattr(self, var_list[modify_index][0], var_list[modify_index][1])
        modify_index += 1
      self.iterate()

  def shape_interactive(self, frameskip):

    print("total nt: {}".format(self.nt))
    frames = range(0, self.nt, frameskip)
    print("total frames: {}".format(frames))
    x2plot = []
    y2plot = []
    in_primary = []
    in_transition = []
    in_secondary = []
    t = []
    max_x, min_x = (0,0)
    max_y, min_y = (0,0)
    for i in frames:
      x = np.insert(sint.cumulative_trapezoid(np.sin(self.A_list[i]), self.s_list[i]), 0, 0)
      y = np.insert(sint.cumulative_trapezoid(np.cos(self.A_list[i]), self.s_list[i]), 0, 0)
      x2plot.append(x)
      y2plot.append(y)
      t.append(i*self.dt)
      in_primary.append(self.s_list[i] > self.s_list[i][-1] - self.lg)
      in_transition.append(np.logical_and(self.s_list[i] > self.s_list[i][-1] - self.lg - self.lt, self.s_list[i] <= self.s_list[i][-1] - self.lg))
      in_secondary.append(self.s_list[i] <= self.s_list[i][-1] - self.lg - self.lt)
      max_x, min_x = (max(max_x, np.max(x)), min(min_x, np.min(x)))
      max_y, min_y = (max(max_y, np.max(y)), min(min_y, np.min(y)))

    w = max_x - min_x
    h = max_y - min_y
    lims = min_x - 0.1*w, max_x + 0.1*w, min_y - 0.1*h, max_y + 0.1*h

    fig = go.Figure()

    for it in range(len(t)):
      fig.add_trace(
        go.Scatter(
          visible=False,
          line=dict(color="red", width=3),
          name="zone 1",
          x = x2plot[it][in_primary[it]],
          y = y2plot[it][in_primary[it]],
          hoverinfo = "skip"
        )
      )
      fig.add_trace(
        go.Scatter(
          visible=False,
          line=dict(color="gray", width=3),
          name="zone 1.5",
          x = np.append(x2plot[it][in_transition[it]], x2plot[it][in_primary[it]][0]),
          y = np.append(y2plot[it][in_transition[it]], y2plot[it][in_primary[it]][0]),
          hoverinfo = "skip"
        )
      )
      fig.add_trace(
        go.Scatter(
          visible=False,
          line=dict(color="blue", width=3),
          name="zone 2",
          x = np.append(x2plot[it][in_secondary[it]], x2plot[it][in_transition[it]][0]),
          y = np.append(y2plot[it][in_secondary[it]], y2plot[it][in_transition[it]][0]),
          hoverinfo = "skip"
        )
      )

    fig.data[0].visible=True
    fig.data[1].visible=True
    fig.data[2].visible=True

    steps = []
    for it in range(len(t)):
      step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
        {"title": "Stem shape at t = {:.3f}".format(t[it])}], label="{:.2f}".format(t[it])
      )
      step["args"][0]["visible"][3*it] = True
      step["args"][0]["visible"][3*it+1] = True
      step["args"][0]["visible"][3*it+2] = True
      steps.append(step)

    sliders = [dict(
      active=0, currentvalue={"prefix": "t = "},
      pad={"t": 20},
      steps=steps
    )]

    fig.update_layout(width=1200,height=1200,
      xaxis=dict(range=[lims[0], lims[1]], autorange=False, zeroline=True),
      yaxis=dict(range=[lims[2], lims[3]], autorange=False, zeroline=False),
      title="Stem shape at t = 0" ,sliders=sliders
    )
    fig.update_yaxes(
      zeroline=True, zerolinewidth=1, zerolinecolor='gray',
      scaleanchor="x",
      scaleratio = 1
    )

    return fig

  def plot_interactive(self, var, frameskip):
    print("total nt: {}".format(self.nt))
    frames = range(0, self.nt, frameskip)
    print("total frames: {}".format(frames))
    x_list = self.s_list
    y_list = getattr(self, var+"_list")
    # nodal attributes
    nodal = len(y_list[0]) == self.ns_list[0]
    x2plot = []
    y2plot = []
    in_primary = []
    in_transition = []
    in_secondary = []
    t = []
    max_x, min_x = (self.s_list[-1][-1], 0)
    max_y, min_y = (0,0)

    for i in frames:
      if nodal:
        x2plot.append(x_list[i])
        in_primary.append(x_list[i] > x_list[i][-1] - self.lg)
        in_transition.append(np.logical_and(x_list[i] > x_list[i][-1] - self.lg - self.lt, x_list[i] <= x_list[i][-1] - self.lg))
        in_secondary.append(x_list[i] <= x_list[i][-1] - self.lg - self.lt)
      else:
        x2plot.append((x_list[i][1:]+x_list[i][:-1])/2)
        in_primary.append(~((x_list[i][1:]+x_list[i][:-1])/2 < x_list[i][-1] - self.lg))
        in_transition.append(np.logical_and((x_list[i][1:]+x_list[i][:-1])/2 > x_list[i][-1] - self.lg - self.lt, (x_list[i][1:]+x_list[i][:-1])/2 <= x_list[i][-1] - self.lg))
        in_secondary.append((x_list[i][1:]+x_list[i][:-1])/2 <= x_list[i][-1] - self.lg - self.lt)
      if var != "theta":
        y2plot.append(y_list[i])
        max_y, min_y = (max(max_y, np.max(y_list[i])), min(min_y, np.min(y_list[i])))
      else:
        y2plot.append(y_list[i]*180/pi)
        max_y, min_y = (max(max_y, np.max(y_list[i]*180/pi)), min(min_y, np.min(y_list[i]*180/pi)))
      t.append(i*self.dt)
    
    w = max_x - min_x
    h = max_y - min_y
    lims = min_x - 0.1*w, max_x + 0.1*w, min_y - 0.1*h, max_y + 0.1*h

    fig = go.Figure()

    for it in range(len(t)):
      fig.add_trace(
        go.Scatter(
          visible=False,
          line=dict(color="red", width=3),
          name="zone 1",
          x = x2plot[it][in_primary[it]],
          y = y2plot[it][in_primary[it]],
          hoverinfo = "skip"
        )
      )
      fig.add_trace(
        go.Scatter(
          visible=False,
          line=dict(color="gray", width=3),
          name="zone 1.5",
          x = np.append(x2plot[it][in_transition[it]], x2plot[it][in_primary[it]][0]),
          y = np.append(y2plot[it][in_transition[it]], y2plot[it][in_primary[it]][0]),
          hoverinfo = "skip"
        )
      )
      fig.add_trace(
        go.Scatter(
          visible=False,
          line=dict(color="blue", width=3),
          name="zone 2",
          x = np.append(x2plot[it][in_secondary[it]], x2plot[it][in_transition[it]][0]),
          y = np.append(y2plot[it][in_secondary[it]], y2plot[it][in_transition[it]][0]),
          hoverinfo = "skip"
        )
      )

    fig.data[0].visible=True
    fig.data[1].visible=True
    fig.data[2].visible=True

    steps = []
    for it in range(len(t)):
      step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
        {"title": "{} at t = {:.3f}".format(var, t[it])}], label="{:.2f}".format(t[it])
      )
      step["args"][0]["visible"][3*it] = True
      step["args"][0]["visible"][3*it+1] = True
      step["args"][0]["visible"][3*it+2] = True
      steps.append(step)

    sliders = [dict(
      active=0, currentvalue={"prefix": "t = "},
      pad={"t": 20},
      steps=steps
    )]

    fig.update_layout(width=1000,height=500,
      xaxis=dict(range=[lims[0], lims[1]], autorange=False, zeroline=True),
      yaxis=dict(range=[lims[2], lims[3]], autorange=False, zeroline=False),
      title="{} at t = 0".format(var) ,sliders=sliders
    )
    fig.update_yaxes(
      zeroline=True, zerolinewidth=1, zerolinecolor='gray'
    )
    
    return fig



