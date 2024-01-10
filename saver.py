import os
import torch
import json
from json import JSONDecodeError
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

# Assists in checkpoint saving and retrieval with a maximum checkpoint count,
# using validation loss or number of iterations
class Saver:
    def __init__(self, model, optimizer, config, epoch_tag = 'epoch_1st'):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.log_dir = config['log_dir']
        self.save_freq = config.get('saver_freq_steps', 500)
        self.save_epoch = config.get('save_freq', 2)
        self.stats_list_name = config.get('saver_stats_list_name',
            'ckpt_stats_list.json')
        self.max_ckpts = config.get('saver_max_ckpts', 5)
        self.save_mode = config.get('saver_mode', 'VAL_LOSS')
        self.stats_list_path = os.path.join(
            self.log_dir, self.stats_list_name)
        self.epoch_tag = epoch_tag

    def cull_to_max_ckpts(self, stats_list, val_loss):
        if self.save_mode == 'VAL_LOSS': # Generally goes down with time
            # Sorted in descending order -- so last elements have lowest loss
            stats_list = sorted(stats_list,
                # not having a recorded val_loss only occurs when we begin training.
                # because we will quickly move past this stage, it will be treated
                # as max loss for purpose of sorting
                key=lambda ckpt: ckpt.get('val_loss', sys.float_info.max) or sys.float_info.max,
                reverse=True)
        elif self.save_mode == 'ITER': # Goes up with time
            # Sorted in ascending order -- so last elements have highest iters
            stats_list = sorted(stats_list,
                key=lambda ckpt: ckpt['iters'])
        need_new = False

        # Always cull nonexistent paths from stats_list
        stats_list = [c for c in stats_list if os.path.exists(
            c.get('save_path', '')) and Path(c.get('save_path', '')
            ).stem.startswith(self.epoch_tag)]

        # if val_loss is None, that means there is no recorded validation loss yet
        if (len(stats_list) >= self.max_ckpts and self.save_mode == 'VAL_LOSS'
            and val_loss is not None):

            # For the purposes of this branch, a None/undefined loss is
            # considered "infinite"
            comp_loss = stats_list[0]['val_loss']
            if comp_loss is None:
                comp_loss = sys.float_info.max

            # If lower than the highest stored loss, we should cull the first
            # element(s) to make room.
            should_cull = val_loss <= comp_loss
            if not should_cull:
                logging.info(f"New checkpoint with val_loss {val_loss}" 
                f" did not outperform worst stored val_loss"
                f" {stats_list[0]['val_loss']}")
                return stats_list, need_new
            need_new = True
            logging.info(f"New best val_loss {val_loss}")

            for cull_ckpt in stats_list[0:len(stats_list) - self.max_ckpts + 1]:
                to_cull_path = cull_ckpt['save_path']
                if os.path.exists(to_cull_path):
                    logging.info(f"Culling {to_cull_path} to make room")
                    os.remove(to_cull_path)

            stats_list = stats_list[len(stats_list) - self.max_ckpts:]
            assert(len(stats_list) == self.max_ckpts)

        # We should use the iter save mode if val_loss is not recorded yet
        elif (len(stats_list) >= self.max_ckpts and self.save_mode == 'ITER'
            or val_loss is None):
            for cull_ckpt in stats_list[0:len(stats_list) - self.max_ckpts + 1]:
                to_cull_path = cull_ckpt['save_path']
                if os.path.exists(to_cull_path):
                    logging.info(f"Culling {to_cull_path} to make room")
                    os.remove(to_cull_path)

            need_new = True
            stats_list = stats_list[len(stats_list) - self.max_ckpts:]

        if len(stats_list) < self.max_ckpts:
            need_new = True

        return stats_list, need_new

    # Retrieves path of most recent checkpoint
    def retrieve_best(self):
        if not os.path.exists(self.log_dir):
            logging.info("No log dir detected.")
            return None

        epoch = self.epoch_tag

        files = list(filter(
            lambda p: p.endswith('.pth') and p.startswith(self.epoch_tag),
            os.listdir(self.log_dir)))

        if not len(files):
            logging.info("No checkpoints detected.")
            return None

        max_ctime = 0
        max_ctime_path = ""
        for f in (os.path.join(self.log_dir, x) for x in files):
            f_time = os.path.getctime(f)
            if (f_time > max_ctime):
                max_ctime = f_time
                max_ctime_path = f
            
        logging.info(f"Saver detects most recent checkpoint {max_ctime_path}")
        return max_ctime_path

    def try_load_stats(self):
        with open(self.stats_list_path) as f:
            try:
                return json.load(f)
            except JSONDecodeError as e:
                logging.warn(f"Failed to decode stats list "
                    f"{self.stats_list_path}, initializing empty list")
                return []

    def try_save(self, epoch, iters, val_loss):
        if not os.path.exists(self.stats_list_path):
            stats_list = []
            with open(self.stats_list_path, 'a') as f:
                json.dump(stats_list, f)
        else:
            stats_list = self.try_load_stats()

        stats_list, need_new = self.cull_to_max_ckpts(stats_list, val_loss)

        if (not need_new):
            return
    
        state = {
            'net': {key: self.model[key].state_dict() for key in self.model},
            'optimizer': self.optimizer.state_dict(),
            'iters': iters,
            'val_loss': val_loss,
            'epoch': epoch
        }
        save_path = os.path.join(self.log_dir, 
            f'{self.epoch_tag}_{epoch}_{hex(iters)[2:]}.pth')
        stats_list.append({
            'iters': iters,
            'epoch': epoch,
            'val_loss': val_loss,
            'save_path': save_path
        })
        logging.info(f'EPOCH {epoch} ITERS {iters} Saving checkpoint to {save_path}')

        with open(self.stats_list_path, 'w') as f:
            json.dump(stats_list, f)
        torch.save(state, save_path)

    def step_hook(self, epoch, iters, val_loss):
        #logging.debug(f'firing step hook with epoch {epoch} iters {iters}')
        if val_loss is not None:
            val_loss = float(val_loss)
        if iters % self.save_freq == 0:
            self.try_save(epoch, iters, val_loss)

    def epoch_hook(self, epoch, iters, val_loss):
        if val_loss is not None:
            val_loss = float(val_loss)
        if epoch % self.save_epoch == 0:
            self.try_save(epoch, iters, val_loss)
