/*******************************************************************************
** CalculateProbability.cpp
** Part of the mutual information toolbox
**
** Contains functions to calculate the probability of each state in the array
** and to calculate the probability of the joint state of two arrays
** 
** Author: Adam Pocock
** Created 17/2/2010
**
**  Copyright 2010 Adam Pocock, The University Of Manchester
**  www.cs.manchester.ac.uk
**
**  This file is part of MIToolbox.
**
**  MIToolbox is free software: you can redistribute it and/or modify
**  it under the terms of the GNU Lesser General Public License as published by
**  the Free Software Foundation, either version 3 of the License, or
**  (at your option) any later version.
**
**  MIToolbox is distributed in the hope that it will be useful,
**  but WITHOUT ANY WARRANTY; without even the implied warranty of
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
**  GNU Lesser General Public License for more details.
**
**  You should have received a copy of the GNU Lesser General Public License
**  along with MIToolbox.  If not, see <http://www.gnu.org/licenses/>.
**
*******************************************************************************/

#include "MIToolbox.h"
#include "ArrayOperations.h"
#include "CalculateProbability.h"
#include "util.h"

JointProbabilityState calculateJointProbability(double *firstVector, double *secondVector, int vectorLength)
{
  int *firstNormalisedVector;
  int *secondNormalisedVector;
  int *firstStateCounts;
  int *secondStateCounts;
  int *jointStateCounts;
  double *firstStateProbs;
  double *secondStateProbs;
  double *jointStateProbs;
  int firstNumStates;
  int secondNumStates;
  int jointNumStates;
  int i;
  double length = vectorLength;
  JointProbabilityState state;

  firstNormalisedVector = safe_calloc(vectorLength,sizeof(int));
  secondNormalisedVector = safe_calloc(vectorLength,sizeof(int));
  
  firstNumStates = normaliseArray(firstVector,firstNormalisedVector,vectorLength);
  secondNumStates = normaliseArray(secondVector,secondNormalisedVector,vectorLength);
  jointNumStates = firstNumStates * secondNumStates;
  
  firstStateCounts = safe_calloc(firstNumStates,sizeof(int));
  secondStateCounts = safe_calloc(secondNumStates,sizeof(int));
  jointStateCounts = safe_calloc(jointNumStates,sizeof(int));
  
  firstStateProbs = safe_calloc(firstNumStates,sizeof(double));
  secondStateProbs =safe_calloc(secondNumStates,sizeof(double));
  jointStateProbs = safe_calloc(jointNumStates,sizeof(double));

	if(firstNormalisedVector == NULL || secondNormalisedVector == NULL ||
		 firstStateCounts ==	 NULL || secondStateCounts == NULL || jointStateCounts == NULL ||
		 firstStateProbs == NULL || secondStateProbs == NULL || jointStateProbs == NULL) {
			
			fprintf(stderr, "could not allocate enough memory");
			exit(EXIT_FAILURE);

		 }

    
  /* optimised version, less numerically stable
  double fractionalState = 1.0 / vectorLength;
  
  for (i = 0; i < vectorLength; i++)
  {
    firstStateProbs[firstNormalisedVector[i]] += fractionalState;
    secondStateProbs[secondNormalisedVector[i]] += fractionalState;
    jointStateProbs[secondNormalisedVector[i] * firstNumStates + firstNormalisedVector[i]] += fractionalState;
  }
  */
  
  /* Optimised for number of FP operations now O(states) instead of O(vectorLength) */
  for (i = 0; i < vectorLength; i++)
  {
    firstStateCounts[firstNormalisedVector[i]] += 1;
    secondStateCounts[secondNormalisedVector[i]] += 1;
    jointStateCounts[secondNormalisedVector[i] * firstNumStates + firstNormalisedVector[i]] += 1;
  }
  
  for (i = 0; i < firstNumStates; i++)
  {
    firstStateProbs[i] = firstStateCounts[i] / length;
  }
  
  for (i = 0; i < secondNumStates; i++)
  {
    secondStateProbs[i] = secondStateCounts[i] / length;
  }
  
  for (i = 0; i < jointNumStates; i++)
  {
    jointStateProbs[i] = jointStateCounts[i] / length;
  }

  FREE_FUNC(firstNormalisedVector);
  FREE_FUNC(secondNormalisedVector);
  FREE_FUNC(firstStateCounts);
  FREE_FUNC(secondStateCounts);
  FREE_FUNC(jointStateCounts);
    
  firstNormalisedVector = NULL;
  secondNormalisedVector = NULL;
  firstStateCounts = NULL;
  secondStateCounts = NULL;
  jointStateCounts = NULL;
  
  /*
  **typedef struct 
  **{
  **  double *jointProbabilityVector;
  **  int numJointStates;
  **  double *firstProbabilityVector;
  **  int numFirstStates;
  **  double *secondProbabilityVector;
  **  int numSecondStates;
  **} JointProbabilityState;
  */
  
  state.jointProbabilityVector = jointStateProbs;
  state.numJointStates = jointNumStates;
  state.firstProbabilityVector = firstStateProbs;
  state.numFirstStates = firstNumStates;
  state.secondProbabilityVector = secondStateProbs;
  state.numSecondStates = secondNumStates;

  return state;
}/*calculateJointProbability(double *,double *, int)*/

ProbabilityState calculateProbability(double *dataVector, int vectorLength)
{
  int *normalisedVector;
  int *stateCounts;
  double *stateProbs;
  int numStates;
  /*double fractionalState;*/
  ProbabilityState state;
  int i;
  double length = vectorLength;

  normalisedVector = safe_calloc(vectorLength,sizeof(int));
  
  numStates = normaliseArray(dataVector,normalisedVector,vectorLength);
  
  stateCounts = safe_calloc(numStates,sizeof(int));
  stateProbs = safe_calloc(numStates,sizeof(double));
  
  /* optimised version, may have floating point problems 
  fractionalState = 1.0 / vectorLength;
  
  for (i = 0; i < vectorLength; i++)
  {
    stateProbs[normalisedVector[i]] += fractionalState;
  }
  */
  
  /* Optimised for number of FP operations now O(states) instead of O(vectorLength) */
  for (i = 0; i < vectorLength; i++)
  {
    stateCounts[normalisedVector[i]] += 1;
  }
  
  for (i = 0; i < numStates; i++)
  {
    stateProbs[i] = stateCounts[i] / length;
  }
  
  FREE_FUNC(stateCounts);
  FREE_FUNC(normalisedVector);
  
  stateCounts = NULL;
  normalisedVector = NULL;
  
  state.probabilityVector = stateProbs;
  state.numStates = numStates;

  return state;
}/*calculateProbability(double *,int)*/

