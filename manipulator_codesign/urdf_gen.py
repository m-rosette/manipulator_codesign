import os
import matplotlib.colors as mcolors
import xml.etree.ElementTree as ET
import tempfile
import random
from xml.dom import minidom
import xml.dom.minidom


class URDFGen:
    def __init__(self, robot_name, save_urdf_dir=None):
        """
        Initializes the URDF generator with the specified robot name and directory to save the URDF file.

        Args:
            robot_name (str): The name of the robot.
            save_urdf_dir (str, optional): The directory where the URDF file will be saved. Defaults to 'urdf/'.
        """
        # TODO: Add a cache clearing method for emptying the tmp/ dir
        self.robot = ET.Element('robot', name=robot_name)
        self.spherical_joint_count = 0  # <--- Track how many have been added

        if save_urdf_dir is None:
            current_working_directory = os.getcwd()
            self.save_urdf_dir = os.path.join(current_working_directory, 'manipulator_codesign/urdf/robots/')
        else:
            self.save_urdf_dir = save_urdf_dir
    
    def create_link(self, name, type='cylinder', link_len=1, link_width=0.05, link_height=None, mass=1.0, origin=[0, 0, 0, 0, 0, 0], material='gray', color=[0.5, 0.5, 0.5, 1], collision=True):#, inertial=True):
        """
        Create a URDF link element.
        
        Args:
            name (str): The name of the link.
            type (str): The type of the shape ('cylinder', 'box', 'sphere'). Default is 'cylinder'.
            link_len (float): The length of the link. Default is 1.
            link_width (float): The width (or radius) of the link. Default is 0.05.
            mass (float): The mass of the link. Default is 1.0.
            origin (list): Origin of the link in the format [x, y, z, roll, pitch, yaw]. Default is [0, 0, 0, 0, 0, 0].
            material (str): Name of the material. Default is 'gray'.
            color (list): RGBA color of the material. Default is [0.5, 0.5, 0.5, 1].
            collision (bool): Whether to include collision element. Default is True.
            inertial (bool): Whether to include inertial element. Default is True.
        
        Returns:
            xml.etree.ElementTree.Element: The created link element.
        """
        link = ET.Element('link', name=name)
        
        inertial = ET.SubElement(link, 'inertial')
        mass = float(mass)
        ET.SubElement(inertial, 'mass', value=str(mass))
        ET.SubElement(inertial, 'inertia', **self.inertial_calculation(mass, link_len, link_width, type))
        ET.SubElement(inertial, 'origin', xyz=' '.join(map(str, origin[:3])), rpy=' '.join(map(str, origin[3:])))
        
        visual = ET.SubElement(link, 'visual')
        ET.SubElement(visual, 'origin', xyz=' '.join(map(str, origin[:3])), rpy=' '.join(map(str, origin[3:])))
        geometry = ET.SubElement(visual, 'geometry')
        material_elem = ET.SubElement(visual, 'material', name=material)
        ET.SubElement(material_elem, 'color', rgba=' '.join(map(str, color)))

        if collision:
            collision_elem = ET.SubElement(link, 'collision')
            ET.SubElement(collision_elem, 'origin', xyz=' '.join(map(str, origin[:3])), rpy=' '.join(map(str, origin[3:])))
            collision_geometry = ET.SubElement(collision_elem, 'geometry')

        shape_elements = {'cylinder': 'cylinder', 'box': 'box', 'sphere': 'sphere'}
        shape_dim = [link_len, link_width]
        if link_height is not None:
            shape_dim.append(link_height)
        if type in shape_elements:
            ET.SubElement(geometry, shape_elements[type], **self._shape_attributes(type, shape_dim))
            if collision:
                ET.SubElement(collision_geometry, shape_elements[type], **self._shape_attributes(type, shape_dim))
        
        return link
    
    def create_joint(self, name, parent, child, origin=[0, 0, 0, 0, 0, 0], joint_type='revolute', axis='0 0 1', limit=[-1.57, 1.57], effort=10, velocity=1):
        """
        Create a joint element for a URDF (Unified Robot Description Format) file.
        Args:
            name (str): The name of the joint.
            parent (str): The name of the parent link.
            child (str): The name of the child link.
            origin (list): The origin of the joint in the format [x, y, z, roll, pitch, yaw]. Default is [0, 0, 0, 0, 0, 0].
            joint_type (str): The type of the joint (e.g., 'revolute', 'prismatic'). Default is 'revolute'.
            axis (str): The axis of rotation or translation in the format 'x y z'. Default is '0 0 1'.
            limit (list): The limits of the joint in the format [lower, upper]. Default is [-1.57, 1.57].
            effort (float): The maximum effort that can be applied by the joint. Default is 10.
            velocity (float): The maximum velocity of the joint. Default is 1.
        Returns:
            xml.etree.ElementTree.Element: The joint element.
        """
        joint = ET.Element('joint', name=name, type=joint_type)
        ET.SubElement(joint, 'parent', link=parent)
        ET.SubElement(joint, 'child', link=child)
        ET.SubElement(joint, 'origin', xyz=' '.join(map(str, origin[:3])), rpy=' '.join(map(str, origin[3:])))
        
        if joint_type in ['revolute', 'prismatic']:
            ET.SubElement(joint, 'axis', xyz=axis)
            ET.SubElement(joint, 'limit', lower=str(limit[0]), upper=str(limit[1]), effort=str(effort), velocity=str(velocity))
        
        return joint
    
    def save_urdf(self, file_name):
        """
        Save the URDF (Unified Robot Description Format) file with a unique name.

        This method ensures that the file name ends with '.urdf' and saves it to the specified directory.
        If a file with the same name already exists, it appends a counter to the file name to make it unique.

        Args:
            file_name (str): The name of the URDF file to be saved.

        Returns:
            None
        """
        if not file_name.endswith('.urdf'):
            file_name += '.urdf'

        if not os.path.exists(self.save_urdf_dir):
            os.makedirs(self.save_urdf_dir)

        file_path = os.path.join(self.save_urdf_dir, file_name)
        base, ext = os.path.splitext(file_path)
        counter = 1
        while os.path.exists(file_path):
            file_path = f"{base}_{counter}{ext}"
            counter += 1

        # Convert ElementTree to a string
        rough_string = ET.tostring(self.robot, encoding='utf-8')

        # Pretty-print using minidom
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="    ")

        # Write the formatted XML to the file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(pretty_xml)

        print(f"\nSaved new urdf to: {file_path}")

    def save_temp_urdf(self):
        """
        Save the URDF (Unified Robot Description Format) file to a temporary file.

        This method saves the URDF file to a temporary location and returns the path of the saved file.

        Returns:
            str: The path of the saved temporary URDF file.
        """
        # Define the directory for temporary URDF files
        tmp_dir = os.path.join(tempfile.gettempdir(), "urdf")
        os.makedirs(tmp_dir, exist_ok=True)  # Ensure the directory exists

        # Generate a unique file name
        tmp_file_path = os.path.join(tmp_dir, next(tempfile._get_candidate_names()) + ".urdf")

        # Convert URDF to pretty XML
        rough_string = ET.tostring(self.robot, encoding="utf-8")
        pretty_string = xml.dom.minidom.parseString(rough_string).toprettyxml(indent="  ")

        # Write to the file
        with open(tmp_file_path, "w", encoding="utf-8") as tmpfile:
            tmpfile.write(pretty_string)
            
        return tmp_file_path
    
    def add_mock_spherical_joint(self, parent, origin, color, color_name, joint_name_prefix="sph", radius=0.06, joint_limits=[-3.14, 3.14]):
        """
        Adds a mock spherical joint composed of three stacked revolute joints (x, y, z axes)
        and a final link with a sphere visual. All joints share the same origin and have no collisions.

        Args:
            parent (str): The name of the link to which the spherical joint attaches.
            joint_name_prefix (str): Prefix for the joint names. Default is "sph".
            sphere_link_name (str): Name of the final link with a visual-only sphere.
            link_radius (float): Radius of the sphere. Default is 0.06.
            joint_limits (list): Rotation limits for each revolute joint. Default is full rotation [-π, π].
        """
        # Axis order: x, y, z
        axes = ['1 0 0', '0 1 0', '0 0 1']

        sphere_joint = []
        for i, axis in enumerate(axes):
            if i != 0:
                origin = [0, 0, 0, 0, 0, 0]

            joint_suffix = f"_{self.spherical_joint_count + 1}"
            joint_name = f"{joint_name_prefix}_joint_{['x', 'y', 'z'][i]}{joint_suffix}"
            child_link = f"{joint_name_prefix}_link_{['x', 'y', 'z'][i]}{joint_suffix}"

            joint = self.create_joint(joint_name, parent, child_link,
                                    origin=origin,
                                    joint_type='revolute',
                                    axis=axis,
                                    limit=joint_limits)
            sphere_joint.append(joint)

            # Add an empty visual-only link with no collision
            link = self.create_link(name=child_link,
                                    type='sphere',
                                    link_len=radius,
                                    link_width=radius,
                                    mass=0.001,
                                    material=color_name,
                                    color=color,
                                    collision=False)
            sphere_joint.append(link)

            parent = child_link  # Stack the next joint on this child      

        self.spherical_joint_count += 1 
            
        return sphere_joint, parent
    
    def create_joint_visual(self, parent_name, parent_length, axis, index, color_name, color_code):
        """
        Creates a small fixed 'cylinder' joint visualization for a revolute joint.

        Args:
            parent_name (str): name of the link to attach the viz to
            parent_length (float): offset along Z from parent link
            axis (str): axis string, e.g. '1 0 0' or '0 1 0'
            index (int): joint index (for naming)
            color_name (str): material/color identifier
            color_code (tuple): RGBA color tuple
        """
        # determine orientation so the cylinder aligns with the revolute axis
        if axis == '1 0 0':
            viz_rot = [0, 1.57, 0]
        elif axis == '0 1 0':
            viz_rot = [1.57, 0, 0]
        else:
            viz_rot = [0, 0, 0]
        origin = [0, 0, 0] + viz_rot

        ball_joint_name      = f'joint_viz{index}'
        ball_joint_link_name = f'joint_viz_link{index}'

        # fixed joint to “attach” the viz to the parent
        self.robot.append(
            self.create_joint(
                name=ball_joint_name,
                parent=parent_name,
                child=ball_joint_link_name,
                joint_type='fixed',
                origin=[0, 0, parent_length, 0, 0, 0]
            )
        )

        # little cylinder representing the joint
        self.robot.append(
            self.create_link(
                name=ball_joint_link_name,
                type='cylinder',
                link_len=0.125,
                link_width=0.06,
                mass=0,
                origin=origin,
                material=color_name,
                color=color_code,
                collision=False
            )
        )
        
    def add_prismatic_joint(self, joint_name, parent_name, child_name, axis, joint_limit, parent_length, child_length, link_width, color_code, color_name, collision, z_buffer=0.01):
        """
        Adds a prismatic (linear) joint to the robot model, optionally with a fixed joint and a slider link for non-z axes.
        Parameters:
            joint_name (str): Name of the prismatic joint to be created.
            parent_name (str): Name of the parent link to which the joint is attached.
            child_name (str): Name of the child link that moves with the joint.
            axis (str): Axis of motion for the prismatic joint, e.g., '0 0 1', '1 0 0', or '0 1 0'.
            joint_limit (list or tuple): Limits of the prismatic joint motion [lower, upper].
            parent_length (float): Length of the parent link (used for joint origin).
            child_length (float): Length of the child link (used for joint limit if axis is '0 0 1').
            link_width (float): Width of the link (used for sizing the slider link).
            color_code (str): Color code for the link material (e.g., hex or RGBA).
            color_name (str): Name of the color/material for the link.
            collision (bool): Whether to add collision geometry to the link.
            z_buffer (float, optional): Buffer distance added to the z-axis for joint limits and origin (helps avoid collisions with previous link). Default is 0.01.
        Notes:
            - For prismatic joints along the z-axis ('0 0 1'), the joint limit and origin are set based on child_length and z_buffer.
            - For prismatic joints along x or y axes, a fixed joint and a slider link are created to represent the linear motion.
            - The function appends the created joints and links to the robot model.
        """
        if axis == '0 0 1':
            joint_limit = [0, child_length - z_buffer]
            origin = [0, 0, link_width + z_buffer, 0, 0, 0]
        
        else:
            # Set the joint_limits based on the child length
            joint_limit = [-child_length/2, child_length/2]

            origin = [0, 0, 0, 0, 0, 0]
            # 1) Create a fixed joint
            joint_name_fixed = f'{joint_name}_fixed'
            child_name_fixed = f'{child_name}_fixed'
            self.robot.append(
                self.create_joint(joint_name_fixed, 
                                parent=parent_name, 
                                child=child_name_fixed, 
                                joint_type='fixed', 
                                axis=axis,
                                limit=joint_limit, 
                                origin=[0, 0, parent_length, 0, 0, 0]))

            # 2) Create fixed 'linear slider' link
            if axis == '1 0 0':
                box_length = abs(joint_limit[0]) + abs(joint_limit[1]) + (2 * link_width)
                box_width = 2 * link_width
                box_height = link_width

            if axis == '0 1 0':
                box_length = 2 * link_width
                box_width = abs(joint_limit[0]) + abs(joint_limit[1]) + (2 * link_width)
                box_height = link_width

            self.robot.append(
                self.create_link(child_name_fixed, 
                                type='box', 
                                link_len=box_length,
                                link_width=box_width,
                                link_height=box_height,
                                origin=[0, 0, -box_height / 2, 0, 0, 0], 
                                material=color_name, 
                                color=color_code, 
                                collision=collision))
            
            parent_name = child_name_fixed

        # 3) Create prismatic joint
        self.robot.append(
            self.create_joint(joint_name, 
                            parent=parent_name, 
                            child=child_name, 
                            joint_type='prismatic', 
                            axis=axis,
                            limit=joint_limit, 
                            origin=origin))

    def add_end_effector_link(self, parent, last_link_length):
        """
        Adds an end effector link to the robot model.

        This method appends a fixed joint and a link representing the end effector
        to the robot's URDF model. The joint connects the specified parent link to
        the end effector link.

        Args:
            parent (str): The name of the parent link to which the end effector will be attached.
            last_link_length (float): The length of the last link in the chain before the end effector.

        Returns:
            None
        """
        # Create a probe end effector
        self.robot.append(self.create_joint('probe_joint', parent, 'probe_link', [0, 0, last_link_length, 0, 0, 0], 'fixed'))
        self.robot.append(self.create_link('probe_link', link_len=0.1, link_width=0.01, mass=0, collision=False))

        self.robot.append(self.create_joint('end_effector_joint', 'probe_link', 'end_effector', [0, 0, 0.1, 0, 0, 0], 'fixed'))
        self.robot.append(self.create_link('end_effector', link_len=0, link_width=0, mass=0, collision=False))
    
    def create_manipulator(self, axes, joint_types, link_lens, joint_lims, link_width=0.05, link_shape='cylinder', collision=False, gripper=False):
        """
        Creates a manipulator robot model with specified parameters.

        Args:
            axes (list): A list of axes for each joint in the manipulator.
            joint_types (list): A list of joint types for each joint in the manipulator.
            link_lens (list): A list of lengths for each link in the manipulator.
            joint_lims (list): A list of joint limits for each joint in the manipulator.
            link_shape (str, optional): The shape of the links. Default is 'cylinder'.
            collision (bool, optional): Whether to enable collision for the links. Default is False.
            gripper (bool, optional): Whether to add a custom gripper at the end of the manipulator. Default is False.

        Returns:
            None
        """
        axes = [self.map_axis_input(axis) if isinstance(axis, int) else axis for axis in axes]
        joint_types = [self.map_joint_type(joint_type) for joint_type in joint_types]
        colors = {name: mcolors.to_rgba(color) for name, color in random.sample(list(mcolors.CSS4_COLORS.items()), len(mcolors.CSS4_COLORS))}
        
        base_link_name = 'base_link'
        base_link_size = 0.25
        self.robot.append(
            self.create_link(name=base_link_name, 
                             type='box', 
                             link_len=base_link_size,
                             link_width=base_link_size,
                             link_height=base_link_size,
                             mass=1,
                             material='gray', 
                             color=[0.5, 0.5, 0.5, 1], 
                             collision=False))
        
        parent_name = base_link_name
        parent_length = base_link_size / 2
        
        for i, (axis, child_length, joint_type, joint_limit) in enumerate(zip(axes, link_lens, joint_types, joint_lims)):
            color_name, color_code = list(colors.items())[i]
            joint_name = f'joint{i}'
            child_name = f'link{i}'

            if joint_type == 'prismatic':
                z_buffer = 0
                if axis == '0 0 1':
                    z_buffer = 0.011
                    child_length = parent_length - link_width + z_buffer

                self.add_prismatic_joint(joint_name, parent_name, child_name, axis, joint_limit, parent_length, child_length, link_width, color_code, color_name, collision, z_buffer)
                
                # child length for x or y translation is a default (short) length since the set child length is used for it's joint_limits
                if axis == '1 0 0' or axis == '0 1 0':
                    child_length = 2 * link_width 

            if joint_type == 'revolute':
                # Create a small fixed 'cylinder' joint visualization for the revolute joint
                self.create_joint_visual(parent_name, parent_length, axis, i, color_name, color_code)

                self.robot.append(
                    self.create_joint(joint_name, 
                                    parent=parent_name, 
                                    child=child_name, 
                                    joint_type=joint_type, 
                                    axis=axis,
                                    limit=joint_limit, 
                                    origin=[0, 0, parent_length, 0, 0, 0]))
            
            if joint_type == 'spherical':
                sphere_radius = 0.1
                spherical_joint, sphere_link = self.add_mock_spherical_joint(
                                                    parent=parent_name,
                                                    radius=sphere_radius,
                                                    origin=[0, 0, parent_length, 0, 0, 0],
                                                    color_name=color_name,
                                                    color=color_code,
                                                    )
                self.robot.extend(spherical_joint)

                # update parent
                parent_name   = sphere_link
                parent_length = sphere_radius

                # now connect the sphere to link{i} with a fixed joint
                self.robot.append(
                    self.create_joint(joint_name,
                                    parent=parent_name,
                                    child=child_name,
                                    origin=[0,0,0,0,0,0],
                                    joint_type='fixed'))

            self.robot.append(
                self.create_link(child_name, 
                                type=link_shape, 
                                link_len=child_length,
                                link_width=link_width,
                                origin=[0, 0, child_length / 2, 0, 0, 0], 
                                material=color_name, 
                                color=color_code, 
                                collision=collision))
        
            parent_name = child_name
            parent_length = child_length
        
        self.add_end_effector_link(parent=parent_name, last_link_length=parent_length)
    
    @staticmethod
    def _shape_attributes(shape_type, shape_dim):
        """
        Generate a dictionary of attributes for a given shape and its dimensions.

        Args:
            shape (str): The type of shape ('cylinder', 'box', or 'sphere').
            shape_dim (list or tuple): The dimensions of the shape. For 'cylinder', 
                                       provide [length, radius]. For 'box', provide 
                                       [length, width, height]. For 'sphere', provide 
                                       [radius].

        Returns:
            dict: A dictionary containing the shape attributes. For 'cylinder', the 
                  dictionary contains 'length' and 'radius'. For 'box', the dictionary 
                  contains 'size'. For 'sphere', the dictionary contains 'radius'. 
                  Returns an empty dictionary if the shape is not recognized.
        """
        if shape_type == 'cylinder':
            return {'length': str(shape_dim[0]), 'radius': str(shape_dim[1])}
        elif shape_type == 'box':
            return {'size': f"{shape_dim[0]} {shape_dim[1]} {shape_dim[2]}"}
        elif shape_type == 'sphere':
            return {'radius': str(shape_dim[0])}
        return {}
    
    @staticmethod
    def map_axis_input(input_axis):
        """
        Maps input axis 0 (x), 1 (y), 2 (z) to '1 0 0', '0 1 0', '0 0 1', respectively.

        Args:
            input_axis (str): The input axis ('x', 'y', 'z').

        Returns:
            str: The corresponding axis in the format 'x y z'.
        """
        axis_mapping = {
            0: '1 0 0',
            1: '0 1 0',
            2: '0 0 1'
        }
        return axis_mapping.get(input_axis, '0 0 0')
    
    @staticmethod
    def map_axis_inverse(vec_str: str) -> int:
        inv = {'1 0 0': 0, '0 1 0': 1, '0 0 1': 2}
        return inv.get(vec_str, 0)
    
    @staticmethod
    def map_joint_type(input_joint_type):
        """
        Maps an input joint type identifier to its corresponding joint type name.

        Args:
            input_joint_type (int): The identifier for the joint type. 
                                    Expected values are:
                                    0 - 'prismatic'
                                    1 - 'revolute'
                                    2 - 'spherical'

        Returns:
            str: The name of the joint type corresponding to the input identifier.
                 Defaults to 'revolute' if the input identifier is not recognized.
        """
        joint_type_mapping = {
            0: 'prismatic',
            1: 'revolute',
            2: 'spherical',
            3: 'fixed'
        }
        return joint_type_mapping.get(input_joint_type, 'revolute')
    
    @staticmethod
    def map_joint_type_inverse(name: str) -> int:
        """
        Maps a joint type name to its corresponding integer code.

        Args:
            name (str): The name of the joint type. 
                Valid values are 'prismatic', 'revolute', 'spherical', and 'fixed'.

        Returns:
            int: The integer code corresponding to the joint type.
                - 0: 'prismatic'
                - 1: 'revolute'
                - 2: 'spherical'
                - 3: 'fixed'
                Defaults to 1 ('revolute') if the name is not recognized.
        """
        inv = {v:k for k,v in {
            0: 'prismatic',
            1: 'revolute',
            2: 'spherical',
            3: 'fixed'
        }.items()}
        return inv.get(name, 1)
    
    @staticmethod
    def inertial_calculation(mass, link_length, link_width, type='cylinder'):
        """
        Calculate the inertia matrix for a given link based on its mass and dimensions.

        Args:
            mass (float): The mass of the link.
            link_length (float): The length of the link.
            link_width (float): The width (or radius) of the link.
            type (str, optional): The type of the link ('cylinder', 'box'). Default is 'cylinder'.

        Returns:
            dict: A dictionary representing the inertia matrix with keys 'ixx', 'iyy', 'izz', 'ixy', 'ixz', 'iyz'.
        """
        if type == 'cylinder':
            ixx = (1/12) * mass * (3 * link_width**2 + link_length**2)
            iyy = (1/12) * mass * (3 * link_width**2 + link_length**2)
            izz = 0.5 * mass * link_width**2
        elif type == 'box':
            ixx = (1/12) * mass * (link_length**2 + link_width**2)
            iyy = (1/12) * mass * (link_length**2 + link_width**2)
            izz = (1/12) * mass * (link_length**2 + link_width**2)
        elif type == 'sphere':
            ixx = (2/5) * mass * link_width**2
            iyy = (2/5) * mass * link_width**2
            izz = (2/5) * mass * link_width**2
        else:
            raise ValueError("Unsupported shape type for inertia calculation.")

        return {
            'ixx': str(ixx), 'iyy': str(iyy), 'izz': str(izz),
            'ixy': '0.0', 'ixz': '0.0', 'iyz': '0.0'  # Explicitly set to string format
        }


if __name__ == '__main__':
    robot_name = 'test_robot'
    urdf_gen = URDFGen(robot_name)

    joint_types = [1, 0, 1, 1]
    axes = [2, 0, 0, 1]
    link_lens = [0.75, 0.25, 0.4, 0.4]

    joint_limit_prismatic = (-0.5, 0.5)
    joint_limit_revolute = (-3.14, 3.14)

    # Map joint limits based on joint type
    joint_limits = [joint_limit_prismatic if jt == 0 else joint_limit_revolute for jt in joint_types]
                   
    urdf_gen.create_manipulator(axes, joint_types, link_lens, link_shape='cylinder', joint_lims=joint_limits, collision=True)
    urdf_gen.save_urdf(robot_name)