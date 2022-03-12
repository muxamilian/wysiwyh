import numpy as np
import queue
import matplotlib.pyplot as plt
import cv2

def plotting_function(q):

  first_time = True

  while True:
    elem = None
    while True:
      try:
        elem = q.get_nowait()
        # print("Got item from queue, was already inside")
      except queue.Empty:
        break
    # print("img", img)
    if elem is None:
      try:
        elem = q.get(timeout=1)
        # print("Got item after waiting")
      except queue.Empty:
        print("Plotting queue empty, exiting plotting_thread")
        quit()

    img = elem[0]
    out_img = cv2.resize(np.clip(elem[2], 0, 1), list(reversed(img.shape[:-1])))
    code = elem[1]
    if first_time:
      fig, axs = plt.subplots(ncols=2,nrows=2)
      fig.set_facecolor("black")
      gs = axs[1, 0].get_gridspec()
      # remove the underlying axes
      for ax in axs[1, :]:
          ax.remove()
      code_ax = fig.add_subplot(gs[1, :])
      plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0);  
      plt.margins(0, 0)
      ax = axs[0,0]
      ax.axis('off')
      ax.set_adjustable('datalim')
      im = ax.imshow(img)
      ax_out = axs[0,1]
      ax_out.axis('off')
      ax_out.set_adjustable('datalim')
      im_out = ax_out.imshow(out_img)
      first_time = False
    else:
      im.set_data(img)
      im_out.set_data(out_img)
    code_ax.clear()
    # code_ax.axis('off')
    code_ax.set_facecolor("black")
    code_ax.bar(x=range(len(code)), height=code, color='white', edgecolor='white')
    plt.pause(0.01)
