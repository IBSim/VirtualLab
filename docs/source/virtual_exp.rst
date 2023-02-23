Virtual Experiments
===================

Tensile Testing
***************

A tensile test is a common test used for mechanical characterisation of materials. A sample of a carefully controlled geometry is gripped from two ends and put under tension. The load can be applied as a controlled force whilst measuring the displacement or as a controlled displacement whilst measuring the required load. This provides information about mechanical properties such as `Young's modulus <https://en.wikipedia.org/wiki/Young%27s_modulus>`_, `Poisson's ratio <https://en.wikipedia.org/wiki/Poisson%27s_ratio>`_, `yield strength <https://en.wikipedia.org/wiki/Yield_strength>`_, and `strain-hardening <https://en.wikipedia.org/wiki/Strain-hardening>`_.

This methodology is so widely used that there exist many international testing standards for various materials and applications. Our initial implementation of this physical experiment as a virtual test is focused on emulating the standard `BS EN ISO 6892-1:2016 <https://www.iso.org/standard/61856.html#:~:text=ISO%206892-1%3A2016>`_ for testing of metallic materials at room temperature and specifically for ‘dog-bone’ shaped samples. The parameterised nature of **VirtualLab** facilitates using our implementation as a template for tensile testing according to other standards or a custom test setup.

To accompany the virtual offering of the platform, the research group have also undertaken a physical experimental campaign with a batch of samples with varying parameters for direct comparison of the experimental and simulation results. This data will be made publicly available in due course.

.. figure:: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/VirtualTesting_01.png
  :width: 600
  :align: center

    Caption

.. figure:: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/VirtualTesting_02.png
  :width: 600
  :align: center

    Caption

.. figure:: https://gitlab.com/ibsim/media/-/raw/master/images/VirtualLab/VirtualTesting_03.png
  :width: 600
  :align: center

    Caption

Laser Flash Analysis
********************

Similarly, Laser flash analysis (LFA) is a commonly used test for thermal characterisation of materials. A disc shaped sample has a short laser pulse incident on one surface, whilst the temperature change is tracked with respect to time on the opposing surface. This is used to measure `thermal diffusivity <https://en.wikipedia.org/wiki/Thermal_diffusivity>`_, which is used to calculate `thermal conductivity <https://en.wikipedia.org/wiki/Thermal_conductivity>`_.

We based our implementation on the testing standards `ASTM E1461 <https://www.astm.org/e1461-13r22.html>`_ / `ASTM E2585 <https://www.astm.org/e2585-09r22.html>`_ for the determination of the thermal diffusivity of primarily homogeneous isotropic solid materials. Other standards can be modelled by varying the parameters of our template.

HIVE
****

Heat by Induction to Verify Extremes (HIVE) is an experimental facility at the `UK Atomic Energy Authority <https://www.gov.uk/government/organisations/uk-atomic-energy-authority>`_’s (UKAEA) `Culham <https://ccfe.ukaea.uk/>`_ site. It is used to expose plasma-facing components to the high thermal loads they will be subjected to in a fusion energy device. In this experiment, samples are thermally loaded on one surface by induction heating whilst being actively cooled with pressurised water. Further information about this custom experiment can be found in this `scientific publication <https://scientific-publications.ukaea.uk/wp-content/uploads/Preprints/UKAEA-CCFE-PR1833.pdf>`_.
