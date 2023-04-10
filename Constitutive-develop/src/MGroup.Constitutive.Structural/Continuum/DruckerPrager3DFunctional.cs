using System;
using System.ComponentModel;

using MGroup.LinearAlgebra;
using MGroup.LinearAlgebra.Matrices;
using MGroup.MSolve.Constitutive;
using MGroup.MSolve.DataStructures;
//using MGroup.Materials.Interfaces;

namespace MGroup.Constitutive.Structural.Continuum
{
	/// <summary>
	///   Class for 3D Drucker Prager material with hardening in functional form.
	/// </summary>
	/// <a href = "https://en.wikipedia.org/wiki/Drucker%E2%80%93Prager_yield_criterion">Wikipedia -Drucker Prager yield criterion</a>
	public class DruckerPrager3DFunctional : IIsotropicContinuumMaterial3D
	{
		/// <summary>
		///   Array for projecting a tensor on its deviatoric part.
		/// </summary>
		private static readonly double[,] DeviatoricProjection = new[,]
		{
			{  2.0/3.0, -1.0/3.0, -1.0/3.0, 0,   0,   0   },
			{ -1.0/3.0,  2.0/3.0, -1.0/3.0, 0,   0,   0   },
			{ -1.0/3.0, -1.0/3.0,  2.0/3.0, 0,   0,   0   },
			{  0,  0,  0, 1.0, 0,   0   },
			{  0,  0,  0, 0,   1.0, 0   },
			{  0,  0,  0, 0,   0,   1.0 }
		};

		/// <summary>
		///   Array for projecting a tensor on its volumetric part.
		/// </summary>
		private static readonly double[,] VolumetricProjection = new[,]
		{
			{ 1.0, 1.0,  1.0, 0,   0,   0   },
			{ 1.0, 1.0,  1.0, 0,   0,   0   },
			{ 1.0, 1.0,  1.0, 0,   0,   0   },
			{  0,  0,  0, 0, 0,   0   },
			{  0,  0,  0, 0,   0, 0   },
			{  0,  0,  0, 0,   0,   0}
		};

		/// <summary>
		///   Identity second order tensor written in vector form.
		/// </summary>
		private static readonly double[] IdentityVector = new[] { 1.0, 1.0, 1.0, 0, 0, 0 };
		private const string EQUIVALENT_STRAIN = "Equivalent strain";
		private const string STRESS_X = "Stress X";
		private const string STRESS_Y = "Stress Y";
		private const string STRESS_Z = "Stress Z";
		private const string STRESS_XY = "Stress XY";
		private const string STRESS_XZ = "Stress XZ";
		private const string STRESS_YZ = "Stress YZ";
		private GenericConstitutiveLawState currentState;

		/// <summary>
		///   The constitutive matrix of the material while still in the elastic region.
		/// </summary>
		private Matrix elasticConstitutiveMatrix = null;

		/// <summary>
		///   Hardening modulus for linear hardening.
		/// </summary>
		private double hardeningModulus;

		/// <summary>
		///   The Poisson ratio.
		/// </summary>
		/// <remarks>
		///   <a href = "http://en.wikipedia.org/wiki/Poisson%27s_ratio">Wikipedia - Poisson's Ratio</a>
		/// </remarks>
		private double poissonRatio;

		/// <summary>
		///   The shear modulus.
		/// </summary>
		/// <remarks>
		///   <a href = "http://en.wikipedia.org/wiki/Shear_modulus">Wikipedia - Shear Modulus</a>
		/// </remarks>
		private readonly double shearModulus;

		/// <summary>
		///   The bulk modulus.
		/// </summary>
		/// <remarks>
		///   <a href = "https://en.wikipedia.org/wiki/Bulk_modulus">Wikipedia - Bulk Modulus</a>
		/// </remarks>
		private readonly double bulkModulus;

		/// <summary>
		///   Initial cohesion/yield stress
		/// </summary>
		private readonly double cohesion;

		/// <summary>
		///   Friction of the material (in rad). Characteristic angle of the yield function
		/// </summary>
		private readonly double friction;

		/// <summary>
		///   Dilation of the material (in rad).  Characteristic angle of the flow potential function
		/// </summary>
		private readonly double dilation;

		private readonly Func<double, double> HardeningFunc;

		private double HardeningGrad;

		/// <summary>
		///   The young modulus.
		/// </summary>
		/// <remarks>
		///   <a href = "http://en.wikipedia.org/wiki/Young%27s_modulus">Wikipedia - Young's Modulus</a>
		/// </remarks>
		private double youngModulus;

		/// <summary>
		///   The constitutive matrix of the material.
		/// </summary>
		private Matrix constitutiveMatrix;

		/// <summary>
		///   The array of incremental strains.
		/// </summary>
		/// <remarks>
		///   <a href = "http://en.wikipedia.org/wiki/Deformation_%28engineering%29">Deformation (engineering)</a>
		/// </remarks>
		private double[] incrementalStrain = new double[6];

		/// <summary>
		///   Indicates whether this <see cref = "IFiniteElementMaterial" /> is modified.
		/// </summary>
		private bool modified;

		/// <summary>
		///   The current strain vector.
		/// </summary>
		private double[] strain = new double[6];

		/// <summary>
		///   The previously converged elastic strain vector.
		/// </summary>
		private double[] strainElasticPrev;

		/// <summary>
		///   The current elastic strain vector.
		/// </summary>
		private double[] strainElastic = new double[6];

		/// <summary>
		///   The previously converged plastic strain vector.
		/// </summary>
		private double[] strainPlasticPrev;

		/// <summary>
		///   The current plastic strain vector.
		/// </summary>
		private double[] strainPlastic = new double[6];

		/// <summary>
		///   The previously converged equivalent/accumulated plastic strain vector.
		/// </summary>
		private double strainEquivalentPrev;

		/// <summary>
		///   The current equivalent/accumulated plastic strain vector.
		/// </summary>
		private double strainEquivalent;

		/// <summary>
		///   The current increment stress vector.
		/// </summary>
		private double[] stresses = new double[6];

		/// <summary>
		///   The current iteration stress vector.
		/// </summary>
		private double[] stressesNew = new double[6];

		/// <summary>
		///   First yield function parameter.
		/// </summary>
		private double heta;

		/// <summary>
		///   Second yield function parameter.
		/// </summary>
		private double ksi;

		/// <summary>
		///   First plastic flow potential function parameter.
		/// </summary>
		private double heta_d;

		/// <summary>
		///   Type of Drucker Prager parameters. 
		///   "outer" for outer Mohr Coulomb approximation(default).
		///   "inner" for inner Mohr Coulomb approximation.
		/// </summary>
		// "a" for Abaqus approximation.
		private string type;

		/// <summary>
		///   Location of the closest projection point on the yield surface. 
		///   For this model there are two possible outcomes: 1)cone 2)apex 
		/// </summary>
		private string yieldPointProjection;

		private bool matrices_not_initialized = true;

		/// <summary>
		///   Initializes a new instance of a hyperbolic <see cref = "DruckerPragerMaterial3D" /> class.
		/// </summary>
		/// <param name = "youngModulus">
		///   The young modulus.
		/// </param>
		/// <param name = "poissonRatio">
		///   The poisson ratio.
		/// </param>
		/// <param name = "friction">
		///   The friction (degrees).
		/// </param>
		/// <param name = "dilation">
		///   The dilation (degrees).
		/// </param>
		/// <param name = "cohesion">
		///   The cohesion/initial yield stress.
		/// </param>
		/// <param name = "hardeningModulus">
		///   The hardening modulus.
		/// <exception cref = "ArgumentException"> When Poisson ratio is equal to 0.5.</exception>
		public DruckerPrager3DFunctional(double youngModulus, double poissonRatio, double friction, double dilation, double cohesion, Func<double, double> hardeningFunc, string type = "outer")
		{
			this.youngModulus = youngModulus;
			this.poissonRatio = poissonRatio;
			this.friction = friction * Math.PI / 180;
			this.dilation = dilation * Math.PI / 180;
			this.cohesion = cohesion;
			this.HardeningFunc = hardeningFunc;

			this.type = type;

			this.shearModulus = this.YoungModulus / (2 * (1 + this.PoissonRatio));
			this.bulkModulus = this.YoungModulus / (3 * (1 - 2 * this.PoissonRatio));
			//double lamda = (this.YoungModulus * this.PoissonRatio) / ((1 + this.PoissonRatio) * (1 - (2 * this.PoissonRatio)));
			//double mi = this.YoungModulus / (2 * (1 + this.PoissonRatio));
			//double value1 = (2 * mi) + lamda;
			DruckerPragerParameters(type);
			elasticConstitutiveMatrix = GetConstitutiveMatrix();
			InitializeMatrices();
		}

		public void InitializeMatrices()
		{
			strainElasticPrev = new double[6];
			strainPlasticPrev = new double[6];
			strainEquivalentPrev = 0;
			constitutiveMatrix = GetConstitutiveMatrix();
			matrices_not_initialized = false;
		}

		private void DruckerPragerParameters(string type)
		{
			if (type == "outer")
			{
				heta = 3 * Math.Sin(friction) / (Math.Sqrt(3) * (3 - Math.Sin(friction)));
				heta_d = 3 * Math.Sin(dilation) / (Math.Sqrt(3) * (3 - Math.Sin(dilation)));
				ksi = 3 * Math.Cos(friction) / (Math.Sqrt(3) * (3 - Math.Sin(friction)));
			}
			else if (type == "inner")
			{
				heta = 6 * Math.Sin(friction) / (Math.Sqrt(3) * (3 + Math.Sin(friction)));
				heta_d = 6 * Math.Sin(dilation) / (Math.Sqrt(3) * (3 + Math.Sin(dilation)));
				ksi = 6 * Math.Cos(friction) / (Math.Sqrt(3) * (3 + Math.Sin(friction)));
			}
			//else if (type == "a")
			//{
			//	heta = Math.Tan(friction) / Math.Sqrt(3);
			//	heta_d = Math.Tan(dilation) / Math.Sqrt(3);
			//	ksi = (3 + Math.Tan(friction)) / (Math.Sqrt(3) * 3);
			//}
			else
			{
				throw new ArgumentException("Given type is not valid");
			}
		}

		public double[] Coordinates { get; set; }

		/// <summary>
		///   Gets the constitutive matrix.
		/// </summary>
		/// <value>
		///   The constitutive matrix.
		/// </value>
		public IMatrixView ConstitutiveMatrix
		{
			get
			{
				if (this.constitutiveMatrix == null) UpdateConstitutiveMatrixAndEvaluateResponse(new double[6]);
				return constitutiveMatrix;
			}
		}

		/// <summary>
		///   Gets the ID of the material.
		/// </summary>
		/// <value>
		///   The id.
		/// </value>
		public int ID => 1;

		/// <summary>
		///   Gets the incremental strains of the finite element's material.
		/// </summary>
		/// <value>
		///   The incremental strains.
		/// </value>
		/// <remarks>
		///   <a href = "http://en.wikipedia.org/wiki/Deformation_%28engineering%29">Deformation (engineering)</a>
		/// </remarks>
		public double[] IncrementalStrains => this.incrementalStrain;

		/// <summary>
		///   Gets a value indicating whether this <see cref = "IFiniteElementMaterial" /> is modified.
		/// </summary>
		/// <value>
		///   <c>true</c> if modified; otherwise, <c>false</c>.
		/// </value>
		public bool Modified => this.modified;

		/// <summary>
		///   Gets the elastic strain.
		/// </summary>
		/// <value>
		///   The elastic strain vector.
		/// </value>
		public double[] StrainElastic => this.strainElastic;

		/// <summary>
		///   Gets the plastic strain.
		/// </summary>
		/// <value>
		///   The plastic strain vector.
		/// </value>
		public double[] StrainPlastic => this.strainPlastic;

		/// <summary>
		///   Gets the Poisson ratio.
		/// </summary>
		/// <value>
		///   The Poisson ratio.
		/// </value>
		/// <remarks>
		///   <a href = "http://en.wikipedia.org/wiki/Poisson%27s_ratio">Wikipedia - Poisson's Ratio</a>
		/// </remarks>
		public double PoissonRatio
		{
			get
			{
				return this.poissonRatio;
			}
			set
			{
				this.poissonRatio = value;
			}
		}

		/// <summary>
		///   Gets the stresses of the finite element's material.
		/// </summary>
		/// <value>
		///   The stresses.
		/// </value>
		/// <remarks>
		///   <a href = "http://en.wikipedia.org/wiki/Stress_%28mechanics%29">Stress (mechanics)</a>
		/// </remarks>
		public double[] Stresses => this.stressesNew;

		/// <summary>
		///   Gets the Young's Modulus.
		/// </summary>
		/// <value>
		///   The young modulus.
		/// </value>
		/// <remarks>
		///   <a href = "http://en.wikipedia.org/wiki/Young%27s_modulus">Wikipedia - Young's Modulus</a>
		/// </remarks>
		public double YoungModulus
		{
			get => this.youngModulus;
			set => this.youngModulus = value;
		}

		public double ShearModulus => shearModulus;

		public double BulkModulus => bulkModulus;

		public double Cohesion => cohesion;

		public double Friction => friction;

		public double Dilation => dilation;

		public double HardeningModulus => hardeningModulus;

		/// <summary>
		///   Creates a new object that is a copy of the current instance.
		/// </summary>
		/// <returns>
		///   A new object that is a copy of this instance.
		/// </returns>
		public object Clone()
		{
			return new DruckerPrager3DFunctional(youngModulus, poissonRatio, friction * 180 / Math.PI, dilation * 180 / Math.PI, cohesion, HardeningFunc, type)
			{
				modified = this.Modified,
				strainEquivalent = this.strainEquivalent,
				strainEquivalentPrev = this.strainEquivalentPrev,
				incrementalStrain = incrementalStrain.Copy(),
				stresses = stresses.Copy()
			};
		}

		/// <summary>
		///   Updates the element's material with the provided incremental strains.
		/// </summary>
		/// <param name = "strainsIncrement">The incremental strains to use for the next step.</param>
		public double[] UpdateConstitutiveMatrixAndEvaluateResponse(double[] strainIncrement)
		{
			if (matrices_not_initialized)
			{ this.InitializeMatrices(); }
			//calculate trial variables
			incrementalStrain.CopyFrom(strainIncrement);
			var strainTrial = new double[incrementalStrain.Length];
			var stressTrial = new double[incrementalStrain.Length];
			var strainDeviatoricTrial = new double[incrementalStrain.Length];
			var stressVolumetricTrial = new double[incrementalStrain.Length];
			var stressDeviatoricTrial = new double[incrementalStrain.Length];
			var normStrainDeviatoricTrial = new double();
			for (int i = 0; i < incrementalStrain.Length; i++)
				strainTrial[i] = strainElasticPrev[i] + incrementalStrain[i];
			for (int i = 0; i < incrementalStrain.Length; i++)
			{
				for (int j = 0; j < incrementalStrain.Length; j++)
				{
					strainDeviatoricTrial[i] = strainDeviatoricTrial[i] + DeviatoricProjection[i, j] * strainTrial[j];
					if (i >= 0 && i <= 2)
						stressTrial[i] += (2 * shearModulus * DeviatoricProjection[i, j] * strainTrial[j] + bulkModulus * VolumetricProjection[i, j] * strainTrial[j]);
					else
						stressTrial[i] += 0.5 * (2 * shearModulus * DeviatoricProjection[i, j] * strainTrial[j] + bulkModulus * VolumetricProjection[i, j] * strainTrial[j]);
				}
			}
			for (int i = 0; i < incrementalStrain.Length; i++)
			{
				for (int j = 0; j < incrementalStrain.Length; j++)
				{
					stressVolumetricTrial[i] += (double)1 / 3 * VolumetricProjection[i, j] * stressTrial[j];
					stressDeviatoricTrial[i] += DeviatoricProjection[i, j] * stressTrial[j];
				}
				normStrainDeviatoricTrial += strainDeviatoricTrial[i] * strainDeviatoricTrial[i];
			}
			normStrainDeviatoricTrial = Math.Sqrt(normStrainDeviatoricTrial);
			var J2 = GetDeviatorSecondStressInvariant(stressTrial);
			bool plasticZone = Math.Sqrt(J2) + heta * stressVolumetricTrial[0] - ksi * HardeningFunc(strainEquivalentPrev) > 0;
			var stressVolumetric = new double[incrementalStrain.Length];
			var stressDeviatoric = new double[incrementalStrain.Length];
			double dlambda = 0;
			if (plasticZone == false)
			{
				stressesNew.CopyFrom(stressTrial);
				stressVolumetric.CopyFrom(stressVolumetricTrial);
				stressDeviatoric.CopyFrom(stressDeviatoricTrial);
				strainEquivalent = strainEquivalentPrev;
				constitutiveMatrix = elasticConstitutiveMatrix.CopyToFullMatrix();
			}
			else
			{
				strainEquivalent = strainEquivalentPrev;
				var yieldFunc = Math.Sqrt(J2) + heta * stressVolumetricTrial[0] - ksi * HardeningFunc(strainEquivalent);
				hardeningModulus = CalculateHardeningGradient(strainEquivalent);
				while (Math.Abs(yieldFunc) >= 1e-6)
				{
					dlambda += yieldFunc / (shearModulus + ksi * ksi * hardeningModulus + bulkModulus * heta * heta_d);
					strainEquivalent = strainEquivalentPrev + ksi * dlambda;
					yieldFunc = Math.Sqrt(J2) - shearModulus * dlambda + heta * (stressVolumetricTrial[0] - bulkModulus * heta_d * dlambda) - ksi * HardeningFunc(strainEquivalent);
					hardeningModulus = CalculateHardeningGradient(strainEquivalent);
				}
				if (dlambda < 0)
					new WarningException("plastic multiplier is negative");
				if (Math.Sqrt(J2) - shearModulus * dlambda >= 0)
				{
					yieldPointProjection = "cone";
					for (int i = 0; i < incrementalStrain.Length; i++)
					{
						stressVolumetric[i] = stressVolumetricTrial[i] - IdentityVector[i] * bulkModulus * heta_d * dlambda;
						stressDeviatoric[i] = (1 - (shearModulus * dlambda) / Math.Sqrt(J2)) * stressDeviatoricTrial[i];
						stressesNew[i] = stressVolumetric[i] + stressDeviatoric[i];
					}
					BuildConsistentConstitutiveMatrix(strainDeviatoricTrial, normStrainDeviatoricTrial, dlambda);
				}
				else
				{
					yieldPointProjection = "apex";
					strainEquivalent = strainEquivalentPrev;
					dlambda = 0;
					var r = HardeningFunc(strainEquivalent) * ksi / heta_d - stressVolumetricTrial[0];
					hardeningModulus = CalculateHardeningGradient(strainEquivalent);
					while (Math.Abs(r) >= 10e-6)
					{
						dlambda += -r / (ksi / heta * ksi / heta_d * hardeningModulus + BulkModulus);
						strainEquivalent = strainEquivalentPrev + ksi / heta * dlambda;
						for (int i = 0; i < incrementalStrain.Length; i++)
						{
							stressVolumetric[i] = stressVolumetricTrial[i] - IdentityVector[i] * bulkModulus * dlambda;
							stressDeviatoric[i] = 0;
							stressesNew[i] = stressVolumetric[i] + stressDeviatoric[i];
						}
						r = HardeningFunc(strainEquivalent) * ksi / heta_d - stressVolumetric[0];
						hardeningModulus = CalculateHardeningGradient(strainEquivalent);
					}

					BuildConsistentConstitutiveMatrix(strainDeviatoricTrial, normStrainDeviatoricTrial, dlambda);
				}
			}
			for (int i = 0; i < incrementalStrain.Length; i++)
			{
				if (i >= 0 && i <= 2)
					strainElastic[i] = 1 / (2 * shearModulus) * stressDeviatoric[i] + 1 / (3 * bulkModulus) * stressVolumetric[i];
				else
					strainElastic[i] = 2 * (1 / (2 * shearModulus) * stressDeviatoric[i] + 1 / (3 * bulkModulus) * stressVolumetric[i]);
				if (dlambda > 0)
					strainPlastic[i] = strainPlasticPrev[i] + strainTrial[i] - strainElastic[i];
				strain[i] = strainElastic[i] + strainPlastic[i];
			}
			return stressesNew;
		}

		/// <summary>
		///   Builds the consistent tangential constitutive matrix.
		/// </summary>
		private void BuildConsistentConstitutiveMatrix(double[] strainDeviatoricTrial, double normStrainDeviatoricTrial, double dlambda)
		{
			if (yieldPointProjection == "cone")
			{
				var A = 1 / (shearModulus + bulkModulus * heta * heta_d + ksi * ksi * hardeningModulus);
				var D = new double[incrementalStrain.Length];
				for (int i = 0; i < incrementalStrain.Length; i++)
				{
					D[i] = strainDeviatoricTrial[i] / normStrainDeviatoricTrial;
				}
				for (int i = 0; i < incrementalStrain.Length; i++)
				{
					for (int j = 0; j < incrementalStrain.Length; j++)
					{
						constitutiveMatrix[i, j] = 2 * shearModulus * (1 - dlambda / (Math.Sqrt(2) * normStrainDeviatoricTrial)) * DeviatoricProjection[i, j]
						+ 2 * shearModulus * (dlambda / (Math.Sqrt(2) * normStrainDeviatoricTrial) - shearModulus * A) * D[i] * D[j] - Math.Sqrt(2) * shearModulus
						* A * bulkModulus * (heta * (D[i] * IdentityVector[j]) + heta_d * (IdentityVector[i] * D[j])) + bulkModulus * (1 - bulkModulus * heta
						* heta_d * A) * VolumetricProjection[i, j];
						if (i > 2 || j > 2)
						{
							constitutiveMatrix[i, j] = 0.5 * constitutiveMatrix[i, j];
						}
					}
				}
			}
			else if (yieldPointProjection == "apex")
			{
				for (int i = 0; i < incrementalStrain.Length; i++)
					for (int j = 0; j < incrementalStrain.Length; j++)
						constitutiveMatrix[i, j] = bulkModulus * (1 - bulkModulus / (bulkModulus + ksi / heta * ksi / heta_d * hardeningModulus)) * VolumetricProjection[i, j];
			}
			else
			{
				throw new ArgumentException();
			}
		}

		/// <summary>
		///   Calculates and returns the first stress invariant (I1).
		/// </summary>
		/// <returns> The first stress invariant (I1).</returns>
		public double GetFirstStressInvariant(double[] stresses) => stresses[0] + stresses[1] + stresses[2];

		/// <summary>
		///   Calculates and returns the mean hydrostatic stress.
		/// </summary>
		/// <returns> The mean hydrostatic stress.</returns>
		public double GetMeanStress(double[] stresses) => GetFirstStressInvariant(stresses) / 3.0;

		/// <summary>
		///   Calculates and returns the second stress invariant (I2).
		/// </summary>
		/// <returns> The second stress invariant (I2).</returns>
		public double GetSecondStressInvariant(double[] stresses)
			=> (stresses[0] * stresses[1]) + (stresses[1] * stresses[2]) + (stresses[0] * stresses[2])
			- Math.Pow(stresses[5], 2) - Math.Pow(stresses[3], 2) - Math.Pow(stresses[4], 2);

		/// <summary>
		///   Calculates and returns the stress deviator tensor in vector form.
		/// </summary>
		/// <returns> The stress deviator tensor in vector form.</returns>
		public double[] GetStressDeviator(double[] stresses)
		{
			var hydrostaticStress = this.GetMeanStress(stresses);
			var stressDeviator = new double[]
			{
			stresses[0] - hydrostaticStress,
			stresses[1] - hydrostaticStress,
			stresses[2] - hydrostaticStress,
			stresses[3],
			stresses[4],
			stresses[5]
			};

			return stressDeviator;
		}

		private double CalculateHardeningGradient(double strain_eq) => HardeningGrad = (HardeningFunc(strain_eq + 1e-8) - HardeningFunc(strain_eq)) / 1e-8;

		/// <summary>
		///   Calculates and returns the third stress invariant (I3).
		/// </summary>
		/// <returns> The third stress invariant (I3). </returns>
		private double GetThirdStressInvariant(double[] stresses)
			=> (stresses[0] * stresses[1] * stresses[2]) + (2 * stresses[5] * stresses[3] * stresses[4])
			- (Math.Pow(stresses[5], 2) * stresses[2]) - (Math.Pow(stresses[3], 2) * stresses[0])
			- (Math.Pow(stresses[4], 2) * stresses[1]);

		/// <summary>
		///   Returns the first stress invariant of the stress deviator tensor (J1), which is zero.
		/// </summary>
		/// <returns> The first stress invariant of the stress deviator tensor (J1). </returns>
		private double GetDeviatorFirstStressInvariant(double[] stresses) => 0;

		/// <summary>
		///   Calculates and returns the second stress invariant of the stress deviator tensor (J2).
		/// </summary>
		/// <returns> The second stress invariant of the stress deviator tensor (J2). </returns>
		private double GetDeviatorSecondStressInvariant(double[] stresses)
		{
			double i1 = this.GetFirstStressInvariant(stresses);
			double i2 = this.GetSecondStressInvariant(stresses);

			double j2 = (1 / 3d * Math.Pow(i1, 2)) - i2;
			return j2;
		}

		/// <summary>
		///   Calculates and returns the third stress invariant of the stress deviator tensor (J3).
		/// </summary>
		/// <returns> The third deviator stress invariant (J3). </returns>
		private double GetDeviatorThirdStressInvariant(double[] stresses)
		{
			double i1 = this.GetFirstStressInvariant(stresses);
			double i2 = this.GetSecondStressInvariant(stresses);
			double i3 = this.GetThirdStressInvariant(stresses);

			double j3 = (2 / 27 * Math.Pow(i1, 3)) - (1 / 3 * i1 * i2) + i3;
			return j3;
		}

		/// <summary>
		///   Clears the stresses of the element's material.
		/// </summary>
		public void ClearStresses()
		{
			stresses.Clear();
			stressesNew.Clear();
		}

		public void ClearState()
		{
			modified = false;
			incrementalStrain.Clear();
			stresses.Clear();
			stressesNew.Clear();
			strain.Clear();
			incrementalStrain.Clear();
			strainElasticPrev.Clear();
			strainElastic.Clear();
			strainPlasticPrev.Clear();
			strainPlastic.Clear();
			strainEquivalent = 0;
			strainEquivalentPrev = 0;
			constitutiveMatrix = GetConstitutiveMatrix();
		}

		/// <summary>
		/// Saves the current stress strain state of the material (after convergence of the iterative solution process
		/// for a given loading step).
		/// </summary>
		public GenericConstitutiveLawState CreateState()
		{
			stresses.CopyFrom(stressesNew);
			strainElasticPrev.CopyFrom(strainElastic);
			strainPlasticPrev.CopyFrom(strainPlastic);
			strainEquivalentPrev = strainEquivalent;
			currentState = new GenericConstitutiveLawState(this, new[]
			{
				(EQUIVALENT_STRAIN, strainEquivalent),
				(STRESS_X, stresses[0]),
				(STRESS_Y, stresses[1]),
				(STRESS_Z, stresses[2]),
				(STRESS_XY, stresses[3]),
				(STRESS_XZ, stresses[4]),
				(STRESS_YZ, stresses[5]),
			});

			return currentState;
		}
		IHaveState ICreateState.CreateState() => CreateState();
		public GenericConstitutiveLawState CurrentState
		{
			get => currentState;
			set
			{
				currentState = value;
				strainEquivalent = currentState.StateValues[EQUIVALENT_STRAIN];
				stresses[0] = currentState.StateValues[STRESS_X];
				stresses[1] = currentState.StateValues[STRESS_Y];
				stresses[2] = currentState.StateValues[STRESS_Z];
				stresses[3] = currentState.StateValues[STRESS_XY];
				stresses[4] = currentState.StateValues[STRESS_XZ];
				stresses[5] = currentState.StateValues[STRESS_YZ];
			}
		}

		/// <summary>
		///   Resets the indicator of whether the material is modified.
		/// </summary>
		public void ResetModified() => this.modified = false;

		private Matrix GetConstitutiveMatrix()
		{
			double fE1 = YoungModulus / (double)(1 + PoissonRatio);
			double fE2 = fE1 * PoissonRatio / (double)(1 - 2 * PoissonRatio);
			double fE3 = fE1 + fE2;
			double fE4 = fE1 * 0.5;
			var afE = Matrix.CreateZero(6, 6);
			afE[0, 0] = fE3;
			afE[0, 1] = fE2;
			afE[0, 2] = fE2;
			afE[1, 0] = fE2;
			afE[1, 1] = fE3;
			afE[1, 2] = fE2;
			afE[2, 0] = fE2;
			afE[2, 1] = fE2;
			afE[2, 2] = fE3;
			afE[3, 3] = fE4;
			afE[4, 4] = fE4;
			afE[5, 5] = fE4;

			return afE;
		}

	}
}
