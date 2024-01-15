import dbdreader
from collections import OrderedDict
import profiles.profiles
from numpy import ones, array
from time import ctime
import ndf

class Slocum(object):
    def __init__(self,filenames,core_parameters=None,extra_parameters={}):
        ''' extra_parameters can be a dictionary with items:
            
            (name_of_parameter, (dbd_parameter_name, unit_after_scaling, scaling_factor))
        '''
        self.filenames=filenames
        self.header=[]
        self.initialise(core_parameters,extra_parameters)

    def initialise(self,core_parameters,extra_parameters):
        parameters=OrderedDict()
        if core_parameters:
            for k,v in core_parameters.items():
                parameters[k]=v
        else:
            parameters["time"]=(None,"s",1.)
            parameters["conductivity"]=("sci_water_cond","mS/cm",10.)
            parameters["temperature"]=("sci_water_temp",'C',1.)
            parameters["pressure"]=("sci_water_pressure","dbar",10)
            parameters["pitch"]=("m_pitch","rad",1.)
            parameters["roll"]=("m_roll","rad",1.)
            parameters["speed"]=("m_speed","m/s",1.)
            parameters["profile_number"]=(None,"-",1)
        for k,v in extra_parameters.items():
            parameters[k]=v
        self.parameters=parameters

    def add_metadata(self,field,mesg):
        self.header.append((field,mesg))

    def convert(self,of=None):
        pass

    def __check_for_ctd_timestamp(self,dbd):
        r=False
        if 'sci_ctd41cp_timestamp' in dbd.parameterNames['sci']:
            t=dbd.get("sci_ctd41cp_timestamp")
            if len(t[1]) and max(t[1])>1e9:
                r=True
        return r

    def __mark_profiles(self,data):
        d=dict(time=data["time"],pressure=data["pressure"])
        p=profiles.profiles.ProfileSplitter(data=d)
        p.split_profiles()
        n_profiles=p.len()
        tmp=ones(data["time"].shape,int)*(-1)
        for i in range(n_profiles):
            tmp[p.i_down[i]]=i*2
            tmp[p.i_up[i]]=i*2+1
        data["profile_number"]=tmp

    def get_data(self):
        dbd=dbdreader.MultiDBD(filenames=self.filenames, include_paired=True)
        has_ctd_time=self.__check_for_ctd_timestamp(dbd)
        parameterlist=[v[0] for k,v in self.parameters.items() if v[0] and k!="sci_water_pressure"]
        if has_ctd_time:
            parameterlist.append("sci_ctd41cp_timestamp")
        x=dbd.get_sync("sci_water_pressure",parameterlist)
        data=dict()
        if has_ctd_time:
            x=x.compress(x[-1]>1e8,axis=1) # remove ctd times equal to
                                           # 0 (and also unrealistic
                                           # CTD values)
            data["time"]=x[-1]
            parameterlist.remove("sci_ctd41cp_timestamp") # not needed anymore
        else:
            data["time"]=x[0]
        data["pressure"]=x[1]
        parameterlist=[k for k,v in self.parameters.items() if v[0] and k!="sci_water_pressure"]
        for i,p in enumerate(parameterlist):
            data[p]=x[i+2]
        dbd.close()
        # filter the data
        #LPF=profiles.filters.LagFilter(1.0,1.73)
        #data["conductivity"]=LPF.filter(data["time"],data["conductivity"])
        #LPF=profiles.filters.LagFilter(1.0,1.37)
        #data["temperature"]=LPF.filter(data["time"],data["temperature"])
        # add the indices for profile identification
        self.__mark_profiles(data)
        return data

class SlocumAscii(Slocum):
    hash="#"
    def __init__(self,filenames,core_parameters={},extra_parameters={}):
        ''' extra_parameters can be a dictionary with items:
            
            (name_of_parameter, (dbd_parameter_name, unit_after_scaling, scaling_factor))
        '''
        Slocum.__init__(self,filenames,core_parameters=core_parameters,extra_parameters=extra_parameters)

    def convert(self,of="output.asc",data=None):
        fd=open(of,'w')
        data=data or self.get_data()
        self.add_metadata("Number of profiles",max(data["profile_number"])+1)
        self.add_metadata("Creation time",ctime())
        self.__write_header(fd)
        self.__write_body(fd,data)
        fd.close()

    def __write_header(self,fd):
        for hdr in self.header:
            s="%s: %s"%hdr
            fd.write("%s %s\n"%(self.hash,s))
        fd.write("%s "%(self.hash))
        for p in list(self.parameters.keys())[:-1]:
            fd.write("%15s"%(p))
        fd.write("%15s\n"%(list(self.parameters.keys())[-1]))
        fd.write("%s "%(self.hash))
        for p in list(self.parameters.values())[:-1]:
            fd.write("%15s"%(p[1]))
        fd.write("%15s\n"%(list(self.parameters.values())[-1][1]))

    def __write_body(self,fd,data):
        # write the data to file
        for j in range(len(data["time"])):
            if data["profile_number"][j]==-1:
                continue
            for p in self.parameters:
                if p=="time":
                    fmt="%15.8f"
                elif p=="profile_number":
                    fmt="%15d"
                else:
                    fmt="%15.8f"
                fd.write(fmt%(data[p][j]*self.parameters[p][2]))
            fd.write("\n")

class SlocumNDF(Slocum):

    def __init__(self,filenames,core_parameters={},extra_parameters={}):
        Slocum.__init__(self,filenames,core_parameters,extra_parameters)

    def convert(self,of="output.ndf"):
        data=self.get_data()
        self.add_metadata("Number of profiles",max(data["profile_number"])+1)
        self.add_metadata("Creation time",ctime())
        self.__write_ndf(data,of)

    def __write_ndf(self,data,of):
        ndf_data=ndf.NDF()
        # adding the data
        parameters_wo_time=dict([(k,v) for k,v in self.parameters.items() if k!='time'])
        for k,v in parameters_wo_time.items():
            # v[2] is the scaling factor
            x=v[2]*data[k]*1.0
            ndf_data.add_parameter(k,v[1],(data["time"],x),str(v[0]))
        # adding meta data
        for k,v in self.header:
            if k=='Creation time':
                continue # to avoid duplicates...
            if k=='Number of profiles':
                # this should be global parameter
                ndf_data.add_global_parameter(k,int(v))
            else:
                ndf_data.add_metadata(k,v)
        ndf_data.save(of)

class Ascii2ndf(object):
    def __init__(self,filename):
        self.filename=filename
        self.initialise_ndf()

    def initialise_ndf(self):
        self.ndf=ndf.NDF()

    def header_to_metadata(self,line):
        line=line.replace(SlocumAscii.hash,"").strip()
        if ":" in line:
            i=line.index(":")
            field,mesg=line[:i],line[i+1:]
            self.ndf.add_metadata(field,mesg)
            
    def add_parameters(self,parameter_info):
        names,units=parameter_info
        names=names.split()
        units=units.split()
        for name,unit in zip(names,units):
            if name=="time":
                continue
            self.ndf.add_parameter(name,unit,(0,0))
        self.names=names

    def read_header(self):
        fp=open(self.filename,'r')
        parameter_info=[]
        while True:
            line=fp.readline()
            if line.startswith(SlocumAscii.hash):
                self.header_to_metadata(line)
                if not ":" in line:
                    tmp=line.replace(SlocumAscii.hash,"").strip()
                    parameter_info.append(tmp)
            else:
                break
        fp.close()
        self.add_parameters(parameter_info)

    def read_body(self):
        fp=open(self.filename,'r')
        data=dict([(k,[]) for k in self.names])
        while True:
            line=fp.readline()
            if not line:
                break
            if line.startswith(SlocumAscii.hash):
                continue
            # we got a data line
            fields=line.split()
            for k,v in zip(self.names,fields):
                data[k].append(float(v))
        fp.close()
        parameters=[n for n in self.names if n!="time"]
        t=array(data["time"])
        for n in parameters:
            self.ndf[n]=(t,array(data[n]))

    def save(self,of="output.ndf"):
        self.ndf.save(of)

        
if __name__=="__main__":
    if 0:
        import glob
        fns=glob.glob("/home/lucas/gliderdata/coconet/hd/sebastian-2013-132-02-08?.[ed]bd")
        s=SlocumAscii(fns)
        s.add_metadata("Originator","me")
        s.convert()
    if 0:
        a=Ascii2ndf('output.asc')
        a.read_header()
        a.read_body()
        a.save()
    if 1:
        import glob
        fns=glob.glob("/home/lucas/gliderdata/coconet/hd/sebastian-2013-132-02-08?.[ed]bd")
        s=SlocumNDF(fns)
        s.add_metadata("Originator","me")
        s.convert()
