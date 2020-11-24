# planning_uncertainty
Planning for Robotics under Uncertainty

# Issues and Solutions
## Deterministic Simulation with Pybullet
Having a deterministic simulation is important for debugging. Pybullet does offer deterministic simulation, but
only if we repeatedly call ```p.resetSimulation()``` before simulating a given (state, action) transition.

## Objects Jumping Out of Ground
Suppose we set an object base(bottom) height to be 0, and we set the ground(plane) height to be 0. We would
expect the object to spawn on the ground and remain stationary. This is not the case with pybullet. What will happen is that the object will seemingly jump out of the ground. In order to address this, we need to always offset 
that object's position by some small amount. In this project, the offset was found experimentally through trial-and-error. This offset is defined in ```Bottle.PLANE_OFFSET``` in the ```sim_ojects.py``` file. In addition, we specify center-of-mass of an object through the ```baseInertialFramePosition``` parameter when spawning a new object. Pybullet internall offsets the base position of objects with this. So if we set an object's position during creation and try querying the simulator for the object's position, we will be returned the base position offset by the center-of-mass, which is undesirable. Nonetheless, this can be addressed by just manually offsetting the returned position by the center-of-mass that we set:

```
object_id = p.createMultiBody(
                baseMass=mass,
                baseInertialFramePosition=center_of_mass_shift,
                baseCollisionShapeIndex=collision_id,
                basePosition=position,
                baseOrientation=orientation)

query_position, query_orientation = p.getBasePositionAndOrientation(object_id)

assert(np.allclose(query_position, position - center_of_mass_shift))
```
