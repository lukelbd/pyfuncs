Configuring proplot
===================

Overview
--------

A special object named `~proplot.rctools.rc`, belonging to the
`~proplot.rctools.rc_configurator` class, is created on import.
This is your one-stop shop for changing global settings belonging to any of
the following three categories:

1. Builtin matplotlib `rcParams <https://matplotlib.org/users/customizing.html>`__
   settings. These have the format ``x.y`` or ``x.y.z``.
2. ProPlot :ref:`rcParamsLong` settings. These also have the format ``x.y``
   (see below).
3. ProPlot :ref:`rcParamsShort` settings. These have no dots (see below).

You can change settings with the `~proplot.rctools.rc` object as follows:

* ``plot.rc.name = value``
* ``plot.rc['name'] = value``
* ``plot.rc.update(name1=value1, name2=value2)``
* ``plot.rc.update({'name1':value1, 'name2':value2})``

To temporarily change settings on a particular axes, use either of the
following:

* ``ax.format(name=value)``
* ``ax.format(rc_kw={'name':value})``

In all of these examples, if the setting name ``name`` contains
any dots, you can simply **omit the dots**. For example, to change the
:rcraw:`title.loc` property, use ``plot.rc.titleloc = value``,
``plot.rc.update(titleloc=value)``, or ``ax.format(titleloc=value)``.

rcParamsShort
-------------

These are **simple, short** names used to change multiple matplotlib and
ProPlot settings at once, as shorthands for settings with longer names, or
for special options. For example, :rcraw:`ticklen` changes the tick length for
the *x* and *y* axes in one go.

.. include:: _static/rcParamsShort.rst

rcParamsLong
------------
These are **longer, specific** setting names
used to customize things not covered by
`~matplotlib.rcParams`.

The ``subplots`` category controls the default layout for figures
and axes. The ``abc``, ``title``, and ``tick`` categories control
a-b-c label, title, and axis tick label settings. The
``suptitle``, ``leftlabel``, ``toplabel``, ``rightlabel``, and ``bottomlabel``
categories control figure title and edge label settings.

There are two new additions to the ``image`` category, and the new
``colorbar`` category controls *inset* and *outer*
`~proplot.axes.Axes.colorbar` properties.
The new ``gridminor`` category controls minor gridline settings,
and the new ``geogrid`` category controls meridian and parallel line settings
for `~proplot.axes.ProjAxes`.

Finally, the ``geoaxes``, ``land``, ``ocean``, ``rivers``, ``lakes``,
``borders``, and ``innerborders`` categories control various
`~proplot.axes.ProjAxes` settings. These are used when the boolean
toggles for the corresponding :ref:`rcParamsShort` settings are turned on.

.. include:: _static/rcParamsLong.rst

The .proplotrc file
-------------------

To modify the global settings, edit your
``~/.proplotrc`` file. To modify settings for a particular project,
create a ``.proplotrc`` file in the same directory as your ipython
notebook, or in an arbitrary parent directory.
As an example, a ``.proplotrc`` file containing the default settings
is shown below. The syntax is mostly the same as the syntax used for
`matplotlibrc files <https://matplotlib.org/3.1.1/tutorials/introductory/customizing.html#customizing-with-matplotlibrc-files>`__.

.. include:: _static/proplotrc
   :literal:
