
import os
import shutil

def set_path(path1, path2=None, task='mkdir', clean=False, ext=None):
   """
      Setup dirs.

      Params:
         - task (string): 'mkdir', 'join', 'check'
         - clean (bool) : True, False 
   """
   if task == 'mkdir':
      if os.path.exists(path1):
         if clean:
            try:
               shutil.rmtree(path1)
            except OSError as e:
               print("Error: %s - %s." % (e.filename, e.strerror))
      else:
         os.mkdir(path1)
      return path1
   
   if task == 'join':
      f = os.path.join(path1, path2)
      if ext is not None:
         f += ext
      return f
   
   if task == 'check':
      if os.path.exists(path1):
         return True
      return False
   