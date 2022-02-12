#pragma once

#include <vector>

// Affiche des proprietes du GPU
void GetGPGPUInfo();

// Initialise la totalite du monde
void initWorld(cudaSurfaceObject_t, cudaSurfaceObject_t, cudaSurfaceObject_t, cudaSurfaceObject_t);

// Libere toutes la memoire alloue sur le GPU
void destroyWorld();

// Genere l'image pour chaque frame et gere la simulation
void GenereImage(cudaSurfaceObject_t, cudaSurfaceObject_t, cudaSurfaceObject_t, cudaSurfaceObject_t, cudaSurfaceObject_t, char *);

// Copie une SurfaceIn dans une surfaceOut
void showOnly(cudaSurfaceObject_t, cudaSurfaceObject_t);

char loadBMPs(FILE *fptrBackground, FILE *fptrRock, FILE *fptrStick, FILE *fptrWater, FILE *fptrFood, FILE *fptrAnthill);

void applyTex2D(cudaSurfaceObject_t surface);