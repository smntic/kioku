{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (ps: with ps; [
      torch
      torchvision
      numpy
      gymnasium
      opencv-python
      pygame
      tensorboard
    ]))
  ];
}
