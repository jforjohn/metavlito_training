import psutil
import pynvml
from os import getpid, getloadavg, cpu_count, name


import warnings
warnings.filterwarnings("ignore")

def get_gpu_used():
    try:
        pynvml.nvmlInit()
        deviceCount = pynvml.nvmlDeviceGetCount()
        meminfo = 0
        utilinfo = 0

        for device_ind in range(deviceCount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_ind)
            meminfo += pynvml.nvmlDeviceGetMemoryInfo(handle).used /1024/1024
            utilinfo += pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        utilinfo /= deviceCount
        pynvml.nvmlShutdown()
    except:
        meminfo = 0
        utilinfo = 0
    return meminfo, utilinfo

def get_cpu_used():
    process = psutil.Process(getpid())
    meminfo = process.memory_info().rss/1024/1024  # in Mbytes
    process.cpu_percent()
    #utilinfo = process.cpu_percent() / psutil.cpu_count()
    if name == 'posix':
        loadinfo = [x / cpu_count() * 100 for x in getloadavg()][1]
    else:
        loadinfo = 0
    return meminfo, loadinfo

"""
from collections import OrderedDict
import json
import subprocess
import sys
import xml.etree.ElementTree
from time import time
import pprint
import os
import subprocess
import gc
import torch

def current_memory_usage():
    '''Returns current memory usage (in MB) of a current process'''
    mem = 0
    if os.name == 'posix':
      out = subprocess.Popen(['ps', '-p', str(os.getpid()), '-o', 'rss'],
                            stdout=subprocess.PIPE).communicate()[0].split(b'\n')
      mem = float(out[1].strip()) / 1024
    return mem

def mem_report():    
  '''
  Report the memory usage of the tensor.storage in pytorch
  Both on CPUs and GPUs are reported
  '''
  start = time()
  #LEN = 65
  #print('='*LEN)
  objects = gc.get_objects()
  tensors = []
  #print('%s\t%s\t\t\t%s' %('Element type', 'Size', 'Used MEM(MBytes)') )
  for obj in objects:
    try:
        if torch.is_tensor(obj) or (
            hasattr(obj, 'data') and torch.is_tensor(obj.data)
            ):
            tensors.append(obj)
    except:
        pass
  cuda_tensors = [t for t in tensors if t.is_cuda]
  host_tensors = [t for t in tensors if not t.is_cuda]
  gpu_mem = _gc_report(cuda_tensors, 'GPU')
  cpu_mem = _gc_report(host_tensors, 'CPU')
  #print('='*LEN)

  return (gpu_mem,  cpu_mem)

def _gc_report(tensors, mem_type):
  '''
  Print the selected tensors of type
  There are two major storage types in our major concern:
      - GPU: tensors transferred to CUDA devices
      - CPU: tensors remaining on the system memory (usually unimportant)
  Args:
      - tensors: the tensors of specified type
      - mem_type: 'CPU' or 'GPU' in current implementation
  '''
  #print('Storage on %s' %(mem_type))
  #print('-'*LEN)
  total_numel = 0
  total_mem = 0
  visited_data = []
  for tensor in tensors:
      if tensor.is_sparse:
          continue
      # a data_ptr indicates a memory block allocated
      data_ptr = tensor.storage().data_ptr()
      if data_ptr in visited_data:
          continue
      visited_data.append(data_ptr)

      numel = tensor.storage().size()
      total_numel += numel
      element_size = tensor.storage().element_size()
      mem = numel*element_size /1024/1024 # 32bit=4Byte, MByte
      total_mem += mem
      '''
      element_type = type(tensor).__name__
      size = tuple(tensor.size())
      print('%s\t\t%s\t\t%.2f' % (
          element_type,
          size,
          mem) )
      '''
  #print('-'*LEN)
  #print('Total Tensors: %d \tUsed Memory Space: %.2f MBytes' % (total_numel, total_mem) )
  #print('-'*LEN)
  return total_mem


from collections import OrderedDict
import json
import subprocess
import sys
import xml.etree.ElementTree

def extract(elem, tag, drop_s):
  text = elem.find(tag).text
  if drop_s not in text: raise Exception(text)
  text = text.replace(drop_s, "")
  try:
    return int(text)
  except ValueError:
    return float(text)

def gpuInfo():
    i = 0

    d = OrderedDict()
    d["time"] = time()

    cmd = ['nvidia-smi', '-q', '-x']
    cmd_out = subprocess.check_output(cmd)
    gpu = xml.etree.ElementTree.fromstring(cmd_out).find("gpu")

    util = gpu.find("utilization")
    d["gpu_util"] = extract(util, "gpu_util", "%")

    d["mem_used"] = extract(gpu.find("fb_memory_usage"), "used", "MiB")
    d["mem_used_per"] = d["mem_used"] * 100 / 11171

    if d["gpu_util"] < 15 and d["mem_used"] < 2816 :
        msg = 'GPU status: Idle \n'
    else:
        msg = 'GPU status: Busy \n'

    #now = time.strftime("%c")
    #print('\n\nUpdated at %s\n\nGPU utilization: %s %%\nVRAM used: %s %%\n\n%s\n\n' % (now, d["gpu_util"],d["mem_used_per"], msg))
    return (d, gpu)


'''
Parse output of nvidia-smi into a python dictionary.
This is very basic!
'''

def parseNvidia():
    sp = subprocess.Popen(['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    out_str = sp.communicate()
    out_list = out_str[0].decode("utf-8").split('\n')

    out_dict = {}

    for item in out_list:
        try:
            key, val = item.split(':')
            key, val = key.strip(), val.strip()
            out_dict[key] = val
        except:
            pass

    return out_dict
"""
