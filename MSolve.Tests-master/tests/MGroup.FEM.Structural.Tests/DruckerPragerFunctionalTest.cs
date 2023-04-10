using MGroup.Constitutive.Structural.Continuum;
using MGroup.LinearAlgebra.Matrices;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace MGroup.FEM.Structural.Tests
{
    public class DruckerPragerFunctionalTest
    {
        [Fact]
        private static void DruckerPragerPlasticityMaterialPointTest()
        {
            //material properties
            var youngModulus = 200.0 * 1e9;
            var poissonRatio = 0.30;
            var friction = 0;
            var dilation = 0;
            var cohesion = 275000000;
            //var hardeningModulus = 2;

            //initialize arrays
            var maxStrain = 0.01;
            var max_strain = new double[6] { 0.005, 0.0, 0.0, 0.005, 0.0, 0.0 };
            //var max_strain = new double[6] { 0.0050, -0.0049, 0.0001, 0.0040, 0.0078, 0.0092 };
            var increments = 10;
            var strain = new double[6];
            var strain1 = new double[increments];
            var stress = new double[increments];
            var C = new IMatrixView[increments];
            var detC = new double[increments];
            for (int j = 0; j < 6; j++)
                strain[j] = max_strain[j] / increments;
            var material = new DruckerPrager3DFunctional(youngModulus, poissonRatio, friction, dilation, cohesion, x => cohesion + 50.0 * 1e9 * x);

            //solve material
            for (int i = 0; i < increments; i++)
            {
                //C[i] = Matrix.CreateZero(6,6); 
                material.UpdateConstitutiveMatrixAndEvaluateResponse(strain);
                strain1[i] = (i + 1) * strain[3];
                stress[i] = material.Stresses[3];
                C[i] = material.ConstitutiveMatrix.CopyToFullMatrix();
                detC[i] = C[i].CopyToFullMatrix().CalcDeterminant();
                material.CreateState();
            }
        }
    }
}
