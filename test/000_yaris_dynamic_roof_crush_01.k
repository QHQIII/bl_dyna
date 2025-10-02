*KEYWORD
$
$
$
$ ===============
$ PARAMETER cards
$ ===============
$
*PARAMETER
R tend           2.2
$
$
$ ===========================
$ DEFINE_TRANSFORMATION cards
$ ===========================
$
*DEFINE_TRANSFORMATION
         1
TRANSL           0.0       0.0       0.0
$
$
$ =============
$ INCLUDE cards
$ =============
$
*INCLUDE_TRANSFORM
YarisD_V2g_ag04_rc05emin.k
         0         0         0         0         0         0         0
         0
       0.0       0.0       0.0                   0
         1
$
*INCLUDE
roof_crush_impactor_03_pos.k
$
*INCLUDE
implicit_control_cards_R9.k
$
$
$
$ =============
$ CONTROL cards
$ =============
$
*CONTROL_IMPLICIT_AUTO
         1        25         5    1.0E-3       -99       0.0         0         0
*CONTROL_IMPLICIT_DYNAMICS
         1      0.55      0.28       0.0       0.0       0.0         2
$
$
$ ============
$ DEFINE cards
$ ============
$
$
*DEFINE_CURVE
$
$: Cross-reference summary for Load-curve 99
$: -----------------------------------------------
$
$: Control card <No label>: Max allowable timestep vs time
$: X axis : Time                       (Units: Time)
$: Y axis : Maximum allowable timestep  (Units: Time)
$
$: Usage: Transient analysis
$
        99         0       0.0       0.0       0.0       0.0         0
                 0.0        5.0000001E-2
           0.2000000        5.0000001E-2
           0.5000000        5.0000001E-2
           1.0000000        2.0000001E-2
           1.1000000        2.5000000E-2
           1.5000000        2.5000000E-2
           2.0000000        2.5000000E-2
           3.0000000        2.5000000E-2
           4.0000000        2.5000000E-2
$
$
*DEFINE_CURVE
$
$: Cross-reference summary for Load-curve 100
$: -----------------------------------------------
$
$: Boundary Prescribed Motion <No label>: Displacement vs time
$: X axis : Time                       (Units: Time)
$: Y axis : Prescribed displacement    (Units: Displacement)
$:    :
$: (To a total of 6 unlabelled Boundary Prescribed Motion entries)
$
$: Usage: Transient analysis
$
       100         0       0.0       0.0       0.0       0.0         0
                 0.0                 0.0
           100.00000                 0.0
$
$
*DEFINE_CURVE
$
$: Cross-reference summary for Load-curve 101
$: -----------------------------------------------
$
$: Boundary Prescribed Motion <No label>: Displacement vs time
$: X axis : Time                       (Units: Time)
$: Y axis : Prescribed displacement    (Units: Displacement)
$
$: Usage: Transient analysis
$
       101         0       0.0       0.0       0.0       0.0         0
                 0.0                 0.0
           0.2000000           2.0000000
           4.0000000           254.00000
           100.00000           254.00000
$
$
*DEFINE_CURVE
$
$: Cross-reference summary for Load-curve 102
$: -----------------------------------------------
$
$: Body load definition <No label>: Base Z acceleration vs time
$: X axis : Time                       (Units: Time)
$: Y axis : Base Z acceleration        (Units: Acceleration)
$
$: Usage: Transient analysis
$
       102         0       0.0       0.0       0.0       0.0         0
                 0.0                 0.0
           1.0000000             1.00000
           4.0000000             1.00000
           100.00000             1.00000
$
$
*DEFINE_CURVE
$
$: Cross-reference summary for Load-curve 110
$: -----------------------------------------------
$
$: Loading definition <No label>: Rigid body force vs time
$: X axis : Time                       (Units: Time)
$: Y axis : Rigid body force           (Units: Force)
$
$: Usage: Transient analysis
$
       110         0       0.0       0.0       0.0       0.0         0
                 0.0                 0.0
           3.0000000           1.0000000
           10.000000           1.0000000
$
$
$
$ ==============
$ BOUNDARY cards
$ ==============
$
*BOUNDARY_PRESCRIBED_MOTION_SET
       100         1         2       100       0.0         0       0.0       0.0
$
$: SET_NODE 100 is in include file YarisD_V2g_ag04_rc01.key
       100         2         2       100       0.0         0       0.0       0.0
$
$: SET_NODE 100 is in include file YarisD_V2g_ag04_rc01.key
       100         3         2       100       0.0         0       0.0       0.0
$
$: SET_NODE 100 is in include file YarisD_V2g_ag04_rc01.key
       100         5         2       100       0.0         0       0.0       0.0
$
$: SET_NODE 100 is in include file YarisD_V2g_ag04_rc01.key
       100         6         2       100       0.0         0       0.0       0.0
$
$: SET_NODE 100 is in include file YarisD_V2g_ag04_rc01.key
       100         7         2       100       0.0         0       0.0       0.0
$
$: SET_NODE 100 is in include file YarisD_V2g_ag04_rc01.key
*BOUNDARY_PRESCRIBED_MOTION_RIGID_LOCAL
         1         1         2       101       0.0         0                 0.0
$
$
$ ==========
$ LOAD cards
$ ==========
$
*LOAD_BODY_Z
       102    9810.0         0
$
$ =============
$ CONTACT cards
$ =============
$
*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE_MORTAR_ID
         1impactor vs roof                                                      
         1         1         3         2         0         0         1         1
       0.1       0.0       0.0       0.0       0.0         0       0.0       0.0
       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0


         3         2
$
*SET_PART_LIST
         1       0.0       0.0       0.0       0.0
   2000173   2000103   2000322   2000323   2000523   2000407
$
$
$: SET_PART 1 is in include file YarisD_V2g_ag04_rc01.key
$
*TITLE
CSSA TOYOTA Yaris implicit roof crush analysis
$
$
*END
