## A story about the baroclinic instability over a sloping bottom

A summary list of functions used for data visualisation in Thesis Chapter 2 and Chapter 3.
- The NetCDF files sourced from post-processing SUNTANS modelling 
- Integrate with some calculation/QAQC functions

### Motivation 
- The confirmed and well-studied mesoscale eddy over the Australian North West Shelf
<img src="Mesoscale_eddy_NWS.jpg" width="400" />

- However, there are unresolved smaller scales dynamics along the coast
<img src="NWS_SST_July.gif" width="500" />

#### Lateral (density) gradients are essential to generate baroclinic instability

A simple example of the formation of lateral gradients 
1. If we have a tank full of water with a __flat__ bottom, and we apply surface cooling

   <img src="baroclinic_instability/IMG_0780.JPG" width="150" />
   <img src="baroclinic_instability/IMG_0781.JPG" width="150" />
   
   With time, we observe the water cools down at the __same rate__
   
   <img src="baroclinic_instability/IMG_0782.JPG" width="150" />

2. If we have a tank full of water with a __sloping__ bottom, and we apply surface cooling
   
   <img src="baroclinic_instability/IMG_0783.JPG" width="150" />
   <img src="baroclinic_instability/IMG_0784.JPG" width="150" />

   With time, we observe the water cools down __faster__ in the shallower region than in the deeper region
   
   <img src="baroclinic_instability/IMG_0785.JPG" width="150" />

   Hence __forms the lateral temperature/density gradients__
   
   <img src="baroclinic_instability/IMG_0786.JPG" width="150" />

   
### Numerical modelling with SUTNANS
- Model set-up
<img src="model_setup.jpg" width="400" />

### Result at a glance
- Eddies observed in the idealised simulation
<img src="eddy_development_NWS.gif" width="300" />

- Schematic diagram of cross-shelf flow
<img src="schematic_diagram_crossshelf_flow.jpg" width="500" />
