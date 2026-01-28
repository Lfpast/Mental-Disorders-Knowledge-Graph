"""
Runner utility for invoking training and evaluation pipelines.

This module exposes a `Runner` class which parses CLI arguments and
invokes `SpERTTrainer.train()` or `SpERTTrainer.eval()` depending on the
`mode` argument.

Inputs:
- An argument list passed to `Runner.run()` (list[str]) resembling
  command-line invocation.

Outputs:
- Side effects: prints progress and launches training/evaluation.
"""

import argparse
from typing import List

##from args import train_argparser, eval_argparser
##from config_reader import process_configs

from spert import input_reader
from spert.spert_trainer import SpERTTrainer

from all_args import get_argparser
from config_reader import read_config_file
import sys


class Runner:
    """High-level orchestrator for training and evaluation runs."""
    
    def __train(self, run_args: argparse.Namespace, config) -> None:
        trainer = SpERTTrainer(run_args, config)
        trainer.train(train_path=run_args.train_path, valid_path=run_args.valid_path,
                types_path=run_args.types_path, input_reader_cls=input_reader.JsonInputReader)

    #def _train():
        #arg_parser = train_argparser()  #Create training-argument parser that adds training-specific arguments
        #process_configs(target=__train, arg_parser=arg_parser) 

    def __eval(self, run_args: argparse.Namespace, config) -> None:
        trainer = SpERTTrainer(run_args, config)
        trainer.eval(dataset_path=run_args.dataset_path, types_path=run_args.types_path,
                input_reader_cls=input_reader.JsonInputReader)


    def run(self, arg_inlist: List[str]) -> None:
        arg_parser = get_argparser()
        args, _ = arg_parser.parse_known_args(arg_inlist)
        # print("*** Parsed commandline arguments: ", args)
        config = read_config_file(args)


        if args.mode == 'train':
            print("="*50)
            print(f"[INFO] Starting Training Pipeline")
            print(f"       Mode: {args.mode}")
            print(f"       Run Name: {args.Names if hasattr(args, 'Names') else args.label}")
            print("="*50)
            #_train()
            self.__train(args, config)
        elif args.mode == 'eval':
            print("="*50)
            print(f"[INFO] Starting Evaluation Pipeline") 
            print("="*50)
            self.__eval(args, config)
        else:
            raise Exception("Mode not in ['train', 'eval'], e.g. 'python spert.py train ...'")


class Runner:
    
     def __train(self, run_args, config):
         trainer = SpERTTrainer(run_args, config)
         trainer.train(train_path=run_args.train_path, valid_path=run_args.valid_path,
                  types_path=run_args.types_path, input_reader_cls=input_reader.JsonInputReader)

     #def _train():
         #arg_parser = train_argparser()  #Create training-argument parser that adds training-specific arguments
         #process_configs(target=__train, arg_parser=arg_parser) 

     def __eval(self, run_args, config):
         trainer = SpERTTrainer(run_args, config)
         trainer.eval(dataset_path=run_args.dataset_path, types_path=run_args.types_path,
                 input_reader_cls=input_reader.JsonInputReader)


     def run(self, arg_inlist):
         arg_parser = get_argparser()
         args, _ = arg_parser.parse_known_args(arg_inlist)
         # print("*** Parsed commandline arguments: ", args)
         config = read_config_file(args)


         if args.mode == 'train':
             print("="*50)
             print(f"[INFO] Starting Training Pipeline")
             print(f"       Mode: {args.mode}")
             print(f"       Run Name: {args.Names if hasattr(args, 'Names') else args.label}")
             print("="*50)
             #_train()
             self.__train(args, config)
         elif args.mode == 'eval':
             print("="*50)
             print(f"[INFO] Starting Evaluation Pipeline") 
             print("="*50)
             self.__eval(args, config)
         else:
             raise Exception("Mode not in ['train', 'eval'], e.g. 'python spert.py train ...'")


