# medit-python
 * medit library python extension
 * 주요 c++ 코드 python

## extension 항목
 * interpolation
   * trillinear-interpolation using openmp
     * clone the repositor
        > cd medit-python/meshlibs/trillinear
        > 
        > python setup.py install
       * test
         * 코드
            ~~~ python
            import pyinterpolate
            pyinterpolate.interpolator.test_main()
            ~~~
         
         * 츨략
            ~~~ 
            tact pybind 0.008508920669555664
            tact scipy interpolator 0.33978724479675293
            faster than x39.93306060690969
            max difference:0.01
            ~~~
   
