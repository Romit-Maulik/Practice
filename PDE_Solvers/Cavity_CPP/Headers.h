#ifndef header_hh_inluded
#define header_hh_inluded

class FVCell // cell class
{
public:
	// Location and index
	double x_loc;
	double y_loc;
	int x_index;
	int y_index;//These are center points of cells only

	// Cell-face velocity components
	double u; // This is the right face technically
	double v; // This is the top face
	double utemp;
	double vtemp;

	// Pressure
	double p;

	// Cell neighbors
	std::shared_ptr< FVCell > xm1;
	std::shared_ptr< FVCell > xp1;
	std::shared_ptr< FVCell > ym1;
	std::shared_ptr< FVCell > yp1;
	
	// Constructor
	FVCell() :
		x_loc( 0.0 ),
		y_loc( 0.0 ),
		x_index( 0 ),
		y_index( 0 ),

		
		// Cell face quantities
		u( 0.0 ),
		v( 0.0 ),
		utemp( 0.0 ),
		vtemp( 0.0 ),

		// Cell center pressure
		p( 0.0 ),
	
		xm1( nullptr ),
		xp1( nullptr ),
		ym1( nullptr ),
		yp1( nullptr )
	{};

	// Destructor
	~FVCell(){};

};

class Global_domain
{
public:

	class Map_2D
	{
	public:
		Map_2D(int i=0, int j=0):
			x_index( i ),
			y_index( j )
		{}

		//Overload operator for equality in the map
		bool operator==(const Map_2D& other) const {
    	return (x_index==other.x_index &&
            y_index==other.y_index);
  		}

		int x_index, y_index;

		// Destructor
		~Map_2D(){};
	};


	// Default constructor
	Global_domain(){};
	// Destructor
	~Global_domain(){};

	struct hasher_key_2D
	{
    	size_t operator()(const Map_2D& map_val) const
    	{
        	return (std::hash<int>()(map_val.x_index) << 1 ^ std::hash<int>()(map_val.y_index));
    	}
	};

	//Virtual function here
	virtual void init_domain()=0;

};


class Cavity_flow : public :: Global_domain
{
public:
	double Re;
	int nx, ny;
	double lid_vel;
	double lside = 1.0;
	double lt = 1.0;
	int nt = 10000;
	double dt = lt/((float)nt);
	double dx, dy;
	double del_p = 0.1;
	std::unordered_map< Map_2D, std::shared_ptr< FVCell >, hasher_key_2D > FVCell_Map;

	// Default constructor
	Cavity_flow() :
		Re( 100.0 ),
		nx (8),
		ny (8),
		lid_vel(2.0)
	{}

	// Member constructor
	Cavity_flow(
		double const Re_,
		int const nx_,
		int const ny_,
		double const lid_vel_
	) :
		Re( Re_ ),
		nx ( nx_ ),
		ny ( ny_ ),
		lid_vel ( lid_vel_ )
	{}

	// Destructor
	~Cavity_flow(){};

	// Member Functions
	void init_domain();
	void update_u();
	void update_v();
	void update_p();
	void artificial_compressibility();
	void plot_results();
};

void Cavity_flow::plot_results()
{
	FILE *fp;
	fp = fopen("Results.txt","wb");

	for(int i=0;i<nx;i++) 
	{
		for(int j=0;j<ny;j++) 
		{
			double u = FVCell_Map[Map_2D(i,j)]->u;
			double v = FVCell_Map[Map_2D(i,j)]->v;
			double p = FVCell_Map[Map_2D(i,j)]->p;
			fprintf(fp,"%f    %f    %f", u , v , p);
			fprintf(fp,"\n");
		}
	}
};

void Cavity_flow::init_domain() // Initialize domain correctly for each cavity_flow class run
{
	//new FVCell (s) are made and pushed to unordered_map for each i,j - This is a 2D Domain
	//New map_key, new FVCell and push both into Cellvect (unordered_map)
	//https://stackoverflow.com/questions/17016175/c-unordered-map-using-a-custom-class-type-as-the-key

	dx = lside/((float)nx);
	dy = lside/((float)nx);

	for (int j=0;j<ny;j++)
	{
		for (int i=0; i<nx; i++)
		{
			std::shared_ptr< FVCell > thisCell( new FVCell );

			thisCell->x_index = i;
			thisCell->y_index = j;

			FVCell_Map[Map_2D(i,j)]=thisCell;
		}
	}

	//Connect neighbors in interior
	for (int j=1;j<ny-1;j++)
	{
		for (int i=1; i<nx-1; i++)
		{
			FVCell_Map[Map_2D(i,j)]->xm1 = FVCell_Map[Map_2D(i-1,j)];
			FVCell_Map[Map_2D(i,j)]->xp1 = FVCell_Map[Map_2D(i+1,j)];
			FVCell_Map[Map_2D(i,j)]->ym1 = FVCell_Map[Map_2D(i,j-1)];
			FVCell_Map[Map_2D(i,j)]->yp1 = FVCell_Map[Map_2D(i,j+1)];
			//Null pointers for z-direction
		}
	}

	// Allocate null-pointers to the boundaries
	//Connect neighbors in boundary - in x
	for (int j=1;j<ny-1;j++)
	{
		FVCell_Map[Map_2D(0,j)]->xm1 = nullptr;
		FVCell_Map[Map_2D(0,j)]->xp1 = FVCell_Map[Map_2D(1,j)];

		FVCell_Map[Map_2D(0,j)]->ym1 = FVCell_Map[Map_2D(0,j-1)];
		FVCell_Map[Map_2D(0,j)]->yp1 = FVCell_Map[Map_2D(0,j+1)];

		FVCell_Map[Map_2D(nx-1,j)]->xm1 = FVCell_Map[Map_2D(nx-2,j)];
		FVCell_Map[Map_2D(nx-1,j)]->xp1 = nullptr;

		FVCell_Map[Map_2D(nx-1,j)]->ym1 = FVCell_Map[Map_2D(nx-1,j-1)];
		FVCell_Map[Map_2D(nx-1,j)]->yp1 = FVCell_Map[Map_2D(nx-1,j+1)];

	}

	//Connect neighbors in boundary - in y
	for (int i=1; i<nx-1; i++)
	{
			FVCell_Map[Map_2D(i,0)]->ym1 = nullptr;
			FVCell_Map[Map_2D(i,0)]->yp1 = FVCell_Map[Map_2D(i,1)];

			FVCell_Map[Map_2D(i,0)]->xm1 = FVCell_Map[Map_2D(i-1,0)];
			FVCell_Map[Map_2D(i,0)]->xp1 = FVCell_Map[Map_2D(i+1,0)];

			FVCell_Map[Map_2D(i,ny-1)]->ym1 = FVCell_Map[Map_2D(i,ny-2)];
			FVCell_Map[Map_2D(i,ny-1)]->yp1 = nullptr;

			FVCell_Map[Map_2D(i,ny-1)]->xm1 = FVCell_Map[Map_2D(i-1,ny-1)];
			FVCell_Map[Map_2D(i,ny-1)]->xp1 = FVCell_Map[Map_2D(i+1,ny-1)];
	}

	// Connect corners
	// Bottom left
	FVCell_Map[Map_2D(0,0)]->ym1 = nullptr;
	FVCell_Map[Map_2D(0,0)]->yp1 = FVCell_Map[Map_2D(0,1)];

	FVCell_Map[Map_2D(0,0)]->xm1 = nullptr;
	FVCell_Map[Map_2D(0,0)]->xp1 = FVCell_Map[Map_2D(1,0)];

	// Top left
	FVCell_Map[Map_2D(0,ny-1)]->yp1 = nullptr;
	FVCell_Map[Map_2D(0,ny-1)]->ym1 = FVCell_Map[Map_2D(0,ny-2)];

	FVCell_Map[Map_2D(0,ny-1)]->xm1 = nullptr;
	FVCell_Map[Map_2D(0,ny-1)]->xp1 = FVCell_Map[Map_2D(1,ny-1)];

	// Bottom right
	FVCell_Map[Map_2D(nx-1,0)]->ym1 = nullptr;
	FVCell_Map[Map_2D(nx-1,0)]->yp1 = FVCell_Map[Map_2D(nx-1,1)];

	FVCell_Map[Map_2D(nx-1,0)]->xp1 = nullptr;
	FVCell_Map[Map_2D(nx-1,0)]->xm1 = FVCell_Map[Map_2D(nx-2,0)];

	// Top right
	FVCell_Map[Map_2D(nx-1,ny-1)]->ym1 = FVCell_Map[Map_2D(nx-1,ny-2)];
	FVCell_Map[Map_2D(nx-1,ny-1)]->yp1 = nullptr;

	FVCell_Map[Map_2D(nx-1,ny-1)]->xp1 = nullptr;
	FVCell_Map[Map_2D(nx-1,ny-1)]->xm1 = FVCell_Map[Map_2D(nx-2,ny-1)];

    // All connected now
	// std::cout<<"Chosen cell: "<<FVCell_Map[Map_2D(0,0)]->x_index<<std::endl;	
	// std::cout<<"Right neighbor x index: "<<FVCell_Map[Map_2D(0,0)]->xp1->x_index<<std::endl;
	// std::cout<<"Right neighbor y index: "<<FVCell_Map[Map_2D(0,0)]->xp1->y_index<<std::endl;
	// std::cout<<"Top-Right neighbor x index: "<<FVCell_Map[Map_2D(0,0)]->xp1->yp1->x_index<<std::endl;
	// std::cout<<"Top-Right neighbor y index: "<<FVCell_Map[Map_2D(0,0)]->xp1->yp1->y_index<<std::endl;
};

void Cavity_flow::artificial_compressibility()
{
	int t = 0;

	while(t<nt)
	{
		t++;
		update_u();
		update_v();
		update_p();
	}

	plot_results();

};


void Cavity_flow::update_u()//Do second order accurate u-momentum equation update
{

	// Interior points - at right face of cell - validated
	for (int i=1;i<nx-1;i++)
	{
		for (int j=1; j<ny-1; j++)
		{
			//East and west u velocities - distance of dx between them
			double u = FVCell_Map[Map_2D(i,j)]->u;
			double u_w = FVCell_Map[Map_2D(i,j)]->xm1->u;
			double u_e = FVCell_Map[Map_2D(i,j)]->xp1->u;
			
			// East and west pressures - distance of dx between them
			double p_w = FVCell_Map[Map_2D(i,j)]->p;
			double p_e = FVCell_Map[Map_2D(i,j)]->xp1->p;

			// North and south u velocities - distance of dx between them
			double u_n = FVCell_Map[Map_2D(i,j)]->yp1->u;
			double u_s = FVCell_Map[Map_2D(i,j)]->ym1->u;

			// Calculation of pressure gradient in x
			double dpdx = (p_e - p_w)/dx;

			// Calculation of convective term in x
			double duudx = (u_e*u_e - u_w*u_w)/(2.0*dx);

			// Calculation of Laplacian
			double d2u = 1.0/Re*((u_e-2*u+u_w)/(dx*dx) + (u_n-2*u+u_s)/(dy*dy));

			// Calculation of convective term in y - needs second order interpolation
			u_n = 0.5*(u_n + u);
			u_s = 0.5*(u_s + u);

			double v_n = 0.5*(FVCell_Map[Map_2D(i,j)]->v + FVCell_Map[Map_2D(i,j)]->xp1->v);
			double v_s = 0.5*(FVCell_Map[Map_2D(i,j)]->ym1->v + FVCell_Map[Map_2D(i,j)]->ym1->xp1->v);

			double duvdy = (u_n*v_n - u_s*v_s)/(dy);
			
			// Update of u velocity
			u = u + dt*(-duudx-duvdy-dpdx+d2u);

			FVCell_Map[Map_2D(i,j)]->utemp = u;
		}
	}

	// Left and right boundaries - except bottom and top left corners
	for (int j=1;j<ny-1;j++)
	{
		// Left boundary
		// East and west u velocities - distance of dx between them
		double u = FVCell_Map[Map_2D(0,j)]->u;
		double u_w = 0.0;
		double u_e = FVCell_Map[Map_2D(0,j)]->xp1->u;
		
		// East and west pressures - distance of dx between them
		double p_w = FVCell_Map[Map_2D(0,j)]->p;
		double p_e = FVCell_Map[Map_2D(0,j)]->xp1->p;

		// North and south u velocities - distance of dx between them
		double u_n = FVCell_Map[Map_2D(0,j)]->yp1->u;
		double u_s = FVCell_Map[Map_2D(0,j)]->ym1->u;

		// Calculation of pressure gradient in x
		double dpdx = (p_e - p_w)/dx;

		// Calculation of convective term in x
		double duudx = (u_e*u_e - u_w*u_w)/(2.0*dx);

		// Calculation of Laplacian
		double d2u = 1.0/Re*((u_e-2*u+u_w)/(dx*dx) + (u_n-2*u+u_s)/(dy*dy));

		// Calculation of convective term in y - needs second order interpolation
		u_n = 0.5*(u_n + u);
		u_s = 0.5*(u_s + u);

		double v_n = 0.5*(FVCell_Map[Map_2D(0,j)]->v + FVCell_Map[Map_2D(0,j)]->xp1->v);
		double v_s = 0.5*(FVCell_Map[Map_2D(0,j)]->ym1->v + FVCell_Map[Map_2D(0,j)]->ym1->xp1->v);

		double duvdy = (u_n*v_n - u_s*v_s)/(dy);
		
		// Update of u velocity
		u = u + dt*(-duudx-duvdy-dpdx+d2u);

		FVCell_Map[Map_2D(0,j)]->utemp = u;

		// Right boundary - not needed for u update
		FVCell_Map[Map_2D(nx-1,j)]->utemp = 0.0;//Wall boundary condition

	}

	// Update top and bottom boundaries - except 
	for (int i = 1; i < nx-1; i++)
	{
		// Bottom boundary
		// East and west u velocities - distance of dx between them
		double u = FVCell_Map[Map_2D(i,0)]->u;
		double u_w = FVCell_Map[Map_2D(i,0)]->xm1->u;
		double u_e = FVCell_Map[Map_2D(i,0)]->xp1->u;
		
		// East and west pressures - distance of dx between them
		double p_w = FVCell_Map[Map_2D(i,0)]->p;
		double p_e = FVCell_Map[Map_2D(i,0)]->xp1->p;

		// North and south u velocities - distance of dx between them
		double u_n = FVCell_Map[Map_2D(i,0)]->yp1->u;
		double u_s = -1.0*(FVCell_Map[Map_2D(i,0)]->u);//Wall boundary condition ghostpoint

		// Calculation of pressure gradient in x
		double dpdx = (p_e - p_w)/dx;

		// Calculation of convective term in x
		double duudx = (u_e*u_e - u_w*u_w)/(2.0*dx);

		// Calculation of Laplacian
		double d2u = 1.0/Re*((u_e-2*u+u_w)/(dx*dx) + (u_n-2*u+u_s)/(dy*dy));

		// Calculation of convective term in y - needs second order interpolation
		u_n = 0.5*(u_n + u);
		u_s = 0.0;

		double v_n = 0.5*(FVCell_Map[Map_2D(i,0)]->v + FVCell_Map[Map_2D(i,0)]->xp1->v);
		double v_s = 0.0;

		double duvdy = (u_n*v_n - u_s*v_s)/(dy);
		
		// Update of u velocity
		u = u + dt*(-duudx-duvdy-dpdx+d2u);

		FVCell_Map[Map_2D(i,0)]->utemp = u;

		// Top boundary		
		// East and west u velocities - distance of dx between them
		u = FVCell_Map[Map_2D(i,ny-1)]->u;
		u_w = FVCell_Map[Map_2D(i,ny-1)]->xm1->u;
		u_e = FVCell_Map[Map_2D(i,ny-1)]->xp1->u;
		
		// East and west pressures - distance of dx between them
		p_w = FVCell_Map[Map_2D(i,ny-1)]->p;
		p_e = FVCell_Map[Map_2D(i,ny-1)]->xp1->p;

		// North and south u velocities - distance of dx between them
		u_n = -u+2.0*lid_vel; //Lid boundary condition ghostpoint
		u_s =  FVCell_Map[Map_2D(i,ny-1)]->ym1->u;

		// Calculation of pressure gradient in x
		dpdx = (p_e - p_w)/dx;

		// Calculation of convective term in x
		duudx = (u_e*u_e - u_w*u_w)/(2.0*dx);
		// Calculation of Laplacian
		d2u = 1.0/Re*((u_e-2*u+u_w)/(dx*dx) + (u_n-2*u+u_s)/(dy*dy));

		// Calculation of convective term in y - needs second order interpolation
		u_n = lid_vel;
		u_s = (u + u_s)*0.5;

		v_n = 0.0;
		v_s = 0.5*(FVCell_Map[Map_2D(i,ny-1)]->ym1->v + FVCell_Map[Map_2D(i,ny-1)]->ym1->xp1->v);

		duvdy = (u_n*v_n - u_s*v_s)/(dy);
		
		// Update of u velocity
		u = u + dt*(-duudx-duvdy-dpdx+d2u);

		FVCell_Map[Map_2D(i,ny-1)]->utemp = u;

	}

	// Bottom and top right corners - wall boundaries
	FVCell_Map[Map_2D(nx-1,0)]->utemp = 0.0;
	FVCell_Map[Map_2D(nx-1,ny-1)]->utemp = 0.0;

	// Bottom left corner
	double u = FVCell_Map[Map_2D(0,0)]->u;
	double u_w = 0.0;
	double u_e = FVCell_Map[Map_2D(0,0)]->xp1->u;
	
	// East and west pressures - distance of dx between them
	double p_w = FVCell_Map[Map_2D(0,0)]->p;
	double p_e = FVCell_Map[Map_2D(0,0)]->xp1->p;

	// North and south u velocities - distance of dx between them
	double u_n = FVCell_Map[Map_2D(0,0)]->yp1->u;
	double u_s = -u;//Wall boundary condition ghostpoint

	// Calculation of pressure gradient in x
	double dpdx = (p_e - p_w)/dx;

	// Calculation of convective term in x
	double duudx = (u_e*u_e - u_w*u_w)/(2.0*dx);

	// Calculation of Laplacian
	double d2u = 1.0/Re*((u_e-2*u+u_w)/(dx*dx) + (u_n-2*u+u_s)/(dy*dy));

	// Calculation of convective term in y - needs second order interpolation
	u_n = 0.5*(u_n + u);
	u_s = 0.0;

	double v_n = 0.5*(FVCell_Map[Map_2D(0,0)]->v + FVCell_Map[Map_2D(0,0)]->xp1->v);
	double v_s = 0.0;

	double duvdy = (u_n*v_n - u_s*v_s)/(dy);
	
	// Update of u velocity
	u = u + dt*(-duudx-duvdy-dpdx+d2u);

	FVCell_Map[Map_2D(0,0)]->utemp = u;


	// Top left corner
	u = FVCell_Map[Map_2D(0,ny-1)]->u;
	u_w = 0.0;
	u_e = FVCell_Map[Map_2D(0,ny-1)]->xp1->u;
	
	// East and west pressures - distance of dx between them
	p_w = FVCell_Map[Map_2D(0,ny-1)]->p;
	p_e = FVCell_Map[Map_2D(0,ny-1)]->xp1->p;

	// North and south u velocities - distance of dx between them
	u_n = 2.0*lid_vel-u;//Lid velocity ghost point
	u_s = FVCell_Map[Map_2D(0,ny-1)]->ym1->u;

	// Calculation of pressure gradient in x
	dpdx = (p_e - p_w)/dx;

	// Calculation of convective term in x
	duudx = (u_e*u_e - u_w*u_w)/(2.0*dx);

	// Calculation of Laplacian
	d2u = 1.0/Re*((u_e-2*u+u_w)/(dx*dx) + (u_n-2*u+u_s)/(dy*dy));

	// Calculation of convective term in y - needs second order interpolation
	u_n = lid_vel;
	u_s = (u+u_s)*0.5;

	v_n = 0.0;
	v_s = 0.5*(FVCell_Map[Map_2D(0,ny-1)]->ym1->v + FVCell_Map[Map_2D(0,ny-1)]->ym1->xp1->v);

	duvdy = (u_n*v_n - u_s*v_s)/(dy);
	
	// Update of u velocity
	u = u + dt*(-duudx-duvdy-dpdx+d2u);

	FVCell_Map[Map_2D(0,ny-1)]->utemp = u;
};


void Cavity_flow::update_v()//Do second order accurate v-momentum equation update
{
	//Update interior points
	for (int i = 1; i < nx-1; i++)
	{
		for (int j = 1; j < ny-1; j++)
		{
			//East and west v velocities - distance of dy between them
			double v = FVCell_Map[Map_2D(i,j)]->v;
			double v_w = FVCell_Map[Map_2D(i,j)]->xm1->v;
			double v_e = FVCell_Map[Map_2D(i,j)]->xp1->v;
			
			// North and South pressures - distance of dy between them
			double p_s = FVCell_Map[Map_2D(i,j)]->p;
			double p_n = FVCell_Map[Map_2D(i,j)]->yp1->p;

			// North and south v velocities - distance of dy between them
			double v_n = FVCell_Map[Map_2D(i,j)]->yp1->v;
			double v_s = FVCell_Map[Map_2D(i,j)]->ym1->v;

			// Calculation of pressure gradient in x
			double dpdy = (p_n - p_s)/dy;

			// Calculation of convective term in y
			double dvvdy = (v_n*v_n - v_s*v_s)/(2.0*dy);

			// Calculation of Laplacian
			double d2v = 1.0/Re*((v_e-2*v+v_w)/(dx*dx) + (v_n-2*v+v_s)/(dy*dy));

			// Calculation of convective term in x - needs second order interpolation
			v_e = 0.5*(v_e + v);
			v_w = 0.5*(v_w + v);

			double u_e = 0.5*(FVCell_Map[Map_2D(i,j)]->u + FVCell_Map[Map_2D(i,j)]->yp1->u);
			double u_w = 0.5*(FVCell_Map[Map_2D(i,j)]->xm1->u + FVCell_Map[Map_2D(i,j)]->xm1->yp1->u);

			double dvudx = (u_e*v_e - v_w*u_w)/(dx);
			
			// Update of u velocity
			v = v + dt*(-dvvdy-dvudx-dpdy+d2v);

			FVCell_Map[Map_2D(i,j)]->vtemp = v;
		}
	}

	// Bottom and top boundaries - except bottom left and bottom right corner
	for (int i = 1; i < nx-1; i++)
	{
		// Bottom boundary
		// East and west v velocities - distance of dy between them
		double v = FVCell_Map[Map_2D(i,0)]->v;
		double v_w = FVCell_Map[Map_2D(i,0)]->xm1->v;
		double v_e = FVCell_Map[Map_2D(i,0)]->xp1->v;
		
		// North and South pressures - distance of dy between them
		double p_s = FVCell_Map[Map_2D(i,0)]->p;
		double p_n = FVCell_Map[Map_2D(i,0)]->yp1->p;

		// North and south v velocities - distance of dy between them
		double v_n = FVCell_Map[Map_2D(i,0)]->yp1->v;
		double v_s = 0.0;

		// Calculation of pressure gradient in x
		double dpdy = (p_n - p_s)/dy;

		// Calculation of convective term in y
		double dvvdy = (v_n*v_n - v_s*v_s)/(2.0*dy);

		// Calculation of Laplacian
		double d2v = 1.0/Re*((v_e-2*v+v_w)/(dx*dx) + (v_n-2*v+v_s)/(dy*dy));

		// Calculation of convective term in x - needs second order interpolation
		v_e = 0.5*(v_e + v);
		v_w = 0.5*(v_w + v);

		double u_e = 0.5*(FVCell_Map[Map_2D(i,0)]->u + FVCell_Map[Map_2D(i,0)]->yp1->u);
		double u_w = 0.5*(FVCell_Map[Map_2D(i,0)]->xm1->u + FVCell_Map[Map_2D(i,0)]->xm1->yp1->u);

		double dvudx = (u_e*v_e - v_w*u_w)/(dx);
		
		// Update of v velocity
		v = v + dt*(-dvvdy-dvudx-dpdy+d2v);

		FVCell_Map[Map_2D(i,0)]->vtemp = v;
		// Top boundary
		FVCell_Map[Map_2D(i,ny-1)]->vtemp = 0.0; //Boundary condition		
	}

	// Left and right boundaries - except corners
	for (int j = 1; j < ny-1; j++)
	{
		// Left boundary
		// East and west v velocities - distance of dy between them
		double v = FVCell_Map[Map_2D(0,j)]->v;
		double v_w = -v;//Wall BC
		double v_e = FVCell_Map[Map_2D(0,j)]->xp1->v;
		
		// North and South pressures - distance of dy between them
		double p_s = FVCell_Map[Map_2D(0,j)]->p;
		double p_n = FVCell_Map[Map_2D(0,j)]->yp1->p;

		// North and south v velocities - distance of dy between them
		double v_n = FVCell_Map[Map_2D(0,j)]->yp1->v;
		double v_s = FVCell_Map[Map_2D(0,j)]->ym1->v;

		// Calculation of pressure gradient in x
		double dpdy = (p_n - p_s)/dy;

		// Calculation of convective term in y
		double dvvdy = (v_n*v_n - v_s*v_s)/(2.0*dy);

		// Calculation of Laplacian
		double d2v = 1.0/Re*((v_e-2*v+v_w)/(dx*dx) + (v_n-2*v+v_s)/(dy*dy));

		// Calculation of convective term in x - needs second order interpolation
		v_e = 0.5*(v_e + v);
		v_w = 0.0;

		double u_e = 0.5*(FVCell_Map[Map_2D(0,j)]->u + FVCell_Map[Map_2D(0,j)]->yp1->u);
		double u_w = 0.0;

		double dvudx = (u_e*v_e - v_w*u_w)/(dx);
		
		// Update of v velocity
		v = v + dt*(-dvvdy-dvudx-dpdy+d2v);
		FVCell_Map[Map_2D(0,j)]->vtemp = v;

		// Right boundary
		// East and west v velocities - distance of dy between them
		v = FVCell_Map[Map_2D(nx-1,j)]->v;
		v_w = FVCell_Map[Map_2D(nx-1,j)]->xm1->v;
		v_e = -v; //Wall BC 
		
		// North and South pressures - distance of dy between them
		p_s = FVCell_Map[Map_2D(nx-1,j)]->p;
		p_n = FVCell_Map[Map_2D(nx-1,j)]->yp1->p;

		// North and south v velocities - distance of dy between them
		v_n = FVCell_Map[Map_2D(nx-1,j)]->yp1->v;
		v_s = FVCell_Map[Map_2D(nx-1,j)]->ym1->v;

		// Calculation of pressure gradient in x
		dpdy = (p_n - p_s)/dy;

		// Calculation of convective term in y
		dvvdy = (v_n*v_n - v_s*v_s)/(2.0*dy);

		// Calculation of Laplacian
		d2v = 1.0/Re*((v_e-2*v+v_w)/(dx*dx) + (v_n-2*v+v_s)/(dy*dy));

		// Calculation of convective term in x - needs second order interpolation
		v_e = 0.0;
		v_w = 0.5*(v_w + v);

		u_e = 0.0;
		u_w = 0.5*(FVCell_Map[Map_2D(nx-1,j)]->xm1->u + FVCell_Map[Map_2D(nx-1,j)]->xm1->yp1->u);

		dvudx = (u_e*v_e - v_w*u_w)/(dx);
		
		// Update of v velocity
		v = v + dt*(-dvvdy-dvudx-dpdy+d2v);
		FVCell_Map[Map_2D(nx-1,j)]->vtemp = v;

	}

	// Updating corners
	// Top left and top right
	FVCell_Map[Map_2D(0,ny-1)]->vtemp = 0.0;
	FVCell_Map[Map_2D(nx-1,ny-1)]->vtemp = 0.0;

	// Bottom left
	// East and west v velocities - distance of dy between them
	double v = FVCell_Map[Map_2D(0,0)]->v;
	double v_w = -v;//Wall BC
	double v_e = FVCell_Map[Map_2D(0,0)]->xp1->v;
	
	// North and South pressures - distance of dy between them
	double p_s = FVCell_Map[Map_2D(0,0)]->p;
	double p_n = FVCell_Map[Map_2D(0,0)]->yp1->p;

	// North and south v velocities - distance of dy between them
	double v_n = FVCell_Map[Map_2D(0,0)]->yp1->v;
	double v_s = 0.0;

	// Calculation of pressure gradient in x
	double dpdy = (p_n - p_s)/dy;

	// Calculation of convective term in y
	double dvvdy = (v_n*v_n - v_s*v_s)/(2.0*dy);

	// Calculation of Laplacian
	double d2v = 1.0/Re*((v_e-2*v+v_w)/(dx*dx) + (v_n-2*v+v_s)/(dy*dy));

	// Calculation of convective term in x - needs second order interpolation
	v_e = 0.5*(v_e + v);
	v_w = 0.0;

	double u_e = 0.5*(FVCell_Map[Map_2D(0,0)]->u + FVCell_Map[Map_2D(0,0)]->yp1->u);
	double u_w = 0.0;

	double dvudx = (u_e*v_e - v_w*u_w)/(dx);
	
	// Update of v velocity
	v = v + dt*(-dvvdy-dvudx-dpdy+d2v);	
	FVCell_Map[Map_2D(0,0)]->vtemp = v;


	// Bottom right
	// East and west v velocities - distance of dy between them
	v = FVCell_Map[Map_2D(nx-1,0)]->v;
	v_w = FVCell_Map[Map_2D(nx-1,0)]->xm1->v;
	v_e = -v;//Wall BC
	
	// North and South pressures - distance of dy between them
	p_s = FVCell_Map[Map_2D(nx-1,0)]->p;
	p_n = FVCell_Map[Map_2D(nx-1,0)]->yp1->p;

	// North and south v velocities - distance of dy between them
	v_n = FVCell_Map[Map_2D(nx-1,0)]->yp1->v;
	v_s = 0.0;

	// Calculation of pressure gradient in x
	dpdy = (p_n - p_s)/dy;

	// Calculation of convective term in y
	dvvdy = (v_n*v_n - v_s*v_s)/(2.0*dy);

	// Calculation of Laplacian
	d2v = 1.0/Re*((v_e-2*v+v_w)/(dx*dx) + (v_n-2*v+v_s)/(dy*dy));

	// Calculation of convective term in x - needs second order interpolation
	v_e = 0.0;
	v_w = 0.5*(v+v_w);

	u_e = 0.0;
	u_w = 0.5*(FVCell_Map[Map_2D(nx-1,0)]->xm1->u + FVCell_Map[Map_2D(nx-1,0)]->xm1->yp1->u);

	dvudx = (u_e*v_e - v_w*u_w)/(dx);
	
	// Update of v velocity
	v = v + dt*(-dvvdy-dvudx-dpdy+d2v);	
	FVCell_Map[Map_2D(nx-1,0)]->vtemp = v;

};


void Cavity_flow::update_p()//Do second order accurate pressure correction - artificial compressibility
{

	// Update temporary to permanent first
	for (int j = 0; j < ny; j++)
	{
		for (int i = 0; i < nx; i++)
		{
			FVCell_Map[Map_2D(i,j)]->u = FVCell_Map[Map_2D(i,j)]->utemp;
			FVCell_Map[Map_2D(i,j)]->v = FVCell_Map[Map_2D(i,j)]->vtemp;
		}
	}

	// Interior points
	for (int j = 1; j < ny-1; j++)
	{
		for (int i = 1; i < nx-1; i++)
		{
			double u_e = FVCell_Map[Map_2D(i,j)]->u;
			double u_w = FVCell_Map[Map_2D(i,j)]->xm1->u;
			double v_n = FVCell_Map[Map_2D(i,j)]->v;
			double v_s = FVCell_Map[Map_2D(i,j)]->ym1->v;

			double dudx = (u_e-u_w)/dx;
			double dvdy = (v_n-v_s)/dy;

			double p = FVCell_Map[Map_2D(i,j)]->p; 
			double pnew = -(dudx+dvdy)*dt*del_p + p;
			FVCell_Map[Map_2D(i,j)]->p = pnew;
		}
	}

	// Bottom and top boundaries
	for (int i = 1; i < nx-1; i++)
	{
		// Bottom
		double u_e = FVCell_Map[Map_2D(i,0)]->u;
		double u_w = FVCell_Map[Map_2D(i,0)]->xm1->u;
		double v_n = FVCell_Map[Map_2D(i,0)]->v;
		double v_s = 0.0;

		double dudx = (u_e-u_w)/dx;
		double dvdy = (v_n-v_s)/dy;

		double p = FVCell_Map[Map_2D(i,0)]->p; 
		double pnew = -(dudx+dvdy)*dt*del_p + p;
		FVCell_Map[Map_2D(i,0)]->p = pnew;

		// Top
		u_e = FVCell_Map[Map_2D(i,ny-1)]->u;
		u_w = FVCell_Map[Map_2D(i,ny-1)]->xm1->u;
		v_n = 0.0;
		v_s = FVCell_Map[Map_2D(i,ny-1)]->ym1->v;

		dudx = (u_e-u_w)/dx;
		dvdy = (v_n-v_s)/dy;

		p = FVCell_Map[Map_2D(i,ny-1)]->p; 
		pnew = -(dudx+dvdy)*dt*del_p + p;
		FVCell_Map[Map_2D(i,ny-1)]->p = pnew;
	}

	// Left and right boundaries
	for (int j = 1; j < ny-1; j++)
	{
		// Left
		double u_e = FVCell_Map[Map_2D(0,j)]->u;
		double u_w = 0.0;
		double v_n = FVCell_Map[Map_2D(0,j)]->v;
		double v_s = FVCell_Map[Map_2D(0,j)]->ym1->v;

		double dudx = (u_e-u_w)/dx;
		double dvdy = (v_n-v_s)/dy;

		double p = FVCell_Map[Map_2D(0,j)]->p; 
		double pnew = -(dudx+dvdy)*dt*del_p + p;
		FVCell_Map[Map_2D(0,j)]->p = pnew;

		// Right
		u_e = 0.0;
		u_w = FVCell_Map[Map_2D(nx-1,j)]->xm1->u;
		v_n = FVCell_Map[Map_2D(nx-1,j)]->v;
		v_s = FVCell_Map[Map_2D(nx-1,j)]->ym1->v;

		dudx = (u_e-u_w)/dx;
		dvdy = (v_n-v_s)/dy;

		p = FVCell_Map[Map_2D(nx-1,j)]->p; 
		pnew = -(dudx+dvdy)*dt*del_p + p;
		FVCell_Map[Map_2D(nx-1,j)]->p = pnew;
	}

	// Bottom left corner
	double u_e = FVCell_Map[Map_2D(0,0)]->u;
	double u_w = 0.0;
	double v_n = FVCell_Map[Map_2D(0,0)]->v;
	double v_s = 0.0;

	double dudx = (u_e-u_w)/dx;
	double dvdy = (v_n-v_s)/dy;

	double p = FVCell_Map[Map_2D(0,0)]->p; 
	double pnew = -(dudx+dvdy)*dt*del_p + p;
	FVCell_Map[Map_2D(0,0)]->p = pnew;

	// Top left corner
	u_e = FVCell_Map[Map_2D(0,ny-1)]->u;
	u_w = 0.0;
	v_n = 0.0;
	v_s = FVCell_Map[Map_2D(0,ny-2)]->v;

	dudx = (u_e-u_w)/dx;
	dvdy = (v_n-v_s)/dy;

	p = FVCell_Map[Map_2D(0,ny-1)]->p; 
	pnew = -(dudx+dvdy)*dt*del_p + p;
	FVCell_Map[Map_2D(0,ny-1)]->p = pnew;


	// Top right corner
	u_e = 0.0;
	u_w = FVCell_Map[Map_2D(nx-1,ny-1)]->xm1->u;
	v_n = 0.0;
	v_s = FVCell_Map[Map_2D(nx-1,ny-1)]->ym1->v;

	dudx = (u_e-u_w)/dx;
	dvdy = (v_n-v_s)/dy;

	p = FVCell_Map[Map_2D(nx-1,ny-1)]->p; 
	pnew = -(dudx+dvdy)*dt*del_p + p;
	FVCell_Map[Map_2D(nx-1,ny-1)]->p = pnew;


	// Bottom right corner
	u_e = 0.0;
	u_w = FVCell_Map[Map_2D(nx-1,0)]->xm1->u;
	v_n = FVCell_Map[Map_2D(nx-1,0)]->yp1->v;
	v_s = 0.0;

	dudx = (u_e-u_w)/dx;
	dvdy = (v_n-v_s)/dy;

	p = FVCell_Map[Map_2D(nx-1,0)]->p; 
	pnew = -(dudx+dvdy)*dt*del_p + p;

}


#endif