using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class SB_RobotSchedulerLogEntityConfiguration : IEntityTypeConfiguration<SB_RobotSchedulerLogEntity>
    {
        public void Configure(EntityTypeBuilder<SB_RobotSchedulerLogEntity> builder)
        {
            builder.ToTable("SB_RobotSchedulerLog");


            builder.Property(x => x.TransID)
                .HasColumnType("bigint(19,0)")
                .IsRequired()
            ;

            builder.Property(x => x.LastExecutionTime)
                .HasColumnType("datetime")
                .IsRequired()
            ;

        }
    }
}
