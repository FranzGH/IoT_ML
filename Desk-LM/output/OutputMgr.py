import abc

import os

class OutputMgr(metaclass=abc.ABCMeta):
   
    @abc.abstractmethod
    def saveParams(self, estimator):
        """Save data to the output."""
       
    def checkCreateDSDir(ds_name, algo_name):
        outdir = './out/'
        if os.path.isdir(outdir) == False:
            os.mkdir(outdir)
        outdir = './out/' + ds_name
        if os.path.isdir(outdir) == False:
            os.mkdir(outdir)
        outdir = './out/' + ds_name + '/include/'
        if os.path.isdir(outdir) == False:
            os.mkdir(outdir)
        outdir = './out/' + ds_name + '/include/' + algo_name + '/'
        if os.path.isdir(outdir) == False:
            os.mkdir(outdir)
        return outdir

    def checkCreateGeneralIncludeDir():
        outdir = './out/'
        if os.path.isdir(outdir) == False:
            os.mkdir(outdir)
        outdir = './out/include/'
        if os.path.isdir(outdir) == False:
            os.mkdir(outdir)
        return outdir

    def checkCreateGeneralSourceDir():
        outdir = './out/'
        if os.path.isdir(outdir) == False:
            os.mkdir(outdir)
        outdir = './out/source/'
        if os.path.isdir(outdir) == False:
            os.mkdir(outdir)
        return outdir

    def cleanSIDirs(path):
        import shutil
        if (os.path.exists(path+'/ds/source/')):
            shutil.rmtree(path+'/ds/source/', ignore_errors=True)
        if os.path.isdir(path) == False:
            os.mkdir(path)
        if os.path.isdir(path+'/ds') == False:
            os.mkdir(path+'/ds')
        os.mkdir(path+'/ds/source/')
        shutil.rmtree(path+'/ds/include/', ignore_errors=True)
        os.mkdir(path+'/ds/include/')

    def saveTestingSet(X_test, y_test, estimator, full=True):
        outdir = OutputMgr.checkCreateDSDir(estimator.dataset.name, estimator.nick)

        if full:
            myFile = open(f"{outdir}testing_set.h","w+")
            myFile.write(f"#ifndef TESTINGSET_H\n")
            myFile.write(f"#define TESTINGSET_H\n\n")
        else:
            myFile = open(f"{outdir}minimal_testing_set.h","w+")
            myFile.write(f"#ifndef MINIMAL_TESTINGSET_H\n")
            myFile.write(f"#define MINIMAL_TESTINGSET_H\n\n")
        myFile.write(f"#define N_TEST {y_test.shape[0]}\n\n")
        myFile.write(f"#ifndef N_FEATURE\n")
        myFile.write(f"#define N_FEATURE {X_test.shape[1]}\n")
        myFile.write(f"#endif\n\n")
        myFile.write(f"#ifndef N_ORIG_FEATURE\n")
        myFile.write(f"#define N_ORIG_FEATURE {X_test.shape[1]}\n")
        myFile.write(f"#endif\n\n")
        if estimator.is_regr:
            type_s = 'float'
        else:
            type_s = 'int'
        myFile.write(f"extern {type_s} y_test[N_TEST];\n")
        myFile.write(f"extern float X_test[N_TEST][N_FEATURE];\n")
        
        #
        #if cfg.normalization!=None and cfg.regr and cfg.algo.lower() != 'dt':
        #    saveTestNormalization(myFile)
        #
        
        myFile.write(f"#endif")
        myFile.close()
        outdirI = OutputMgr.checkCreateGeneralIncludeDir()
        from shutil import copyfile
        if full:
            copyfile(f"{outdir}testing_set.h", f"{outdirI}testing_set.h")
        else:
            copyfile(f"{outdir}minimal_testing_set.h", f"{outdirI}minimal_testing_set.h")

        outdirS = OutputMgr.checkCreateGeneralSourceDir()
        if full:
            myFile = open(f"{outdirS}testing_set.c","w+")
        else:
            myFile = open(f"{outdirS}minimal_testing_set.c","w+")
        #myFile.write(f"#include \"AI_main.h\"\n")
        if full:
            myFile.write(f"#include \"testing_set.h\"\n")
        else:
            myFile.write(f"#include \"minimal_testing_set.h\"\n")

        if estimator.is_regr:
            type_s = 'float'
        else:
            type_s = 'int'
        import sys
        sys.path.insert(1, 'utils')
        import create_matrices
        import numpy as np
        stri = create_matrices.createArray(type_s, "y_test", np.reshape(y_test, (y_test.shape[0], )), 'N_TEST')
        myFile.write(stri)
        
        stri = create_matrices.createMatrix('float', 'X_test', X_test.values, 'N_TEST', 'N_FEATURE') # changed by FB
        myFile.write(stri)
        myFile.close()

    def saveTrainingSet(X_train, y_train, estimator):
        outdir = OutputMgr.checkCreateDSDir()

        myFile = open(f"{outdir}training_set.h","w+")
        myFile.write(f"#define N_TRAIN {y_train.shape[0]}\n\n")
        myFile.write(f"#ifndef N_FEATURE\n")
        myFile.write(f"#define N_FEATURE {X_train.shape[1]}\n")
        myFile.write(f"#endif\n\n")

        if estimator.is_regr:
            type_s = 'float'
        else:
            type_s = 'int'
        myFile.write(f"extern {type_s} y_train[N_TRAIN];\n")
        myFile.write(f"extern float X_train[N_TRAIN][N_FEATURE];\n")
        myFile.close()
        
        outdirI = OutputMgr.checkCreateGeneralIncludeDir()
        from shutil import copyfile
        copyfile(f"{outdir}training_set.h", f"{outdirI}training_set.h")

        outdirS = OutputMgr.checkCreateGeneralSourceDir()
        myFile = open(f"{outdirS}training_set_params.c","w+")
        #myFile.write(f"#include \"AI_main.h\"\n")
        myFile.write(f"#include \"training_set.h\"\n")